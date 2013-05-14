#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "SkeinTest.h"

#define TILE_SIZE 256

// Fast version of 64-bit rotate left for GPU
#define RotL_64_GPU(x,N) (((x) << (N)) | ((x) >> (64-(N))))

/* macro to perform a key injection (same for all block sizes) */
#define InjectKey_GPU(r)                                            \
    for (i=0;i < WCNT;i++)                                          \
         X[i] += ks[((r)+i) % (WCNT+1)];                            \
    X[WCNT-3] += ts[((r)+0) % 3];                                   \
    X[WCNT-2] += ts[((r)+1) % 3];                                   \
    X[WCNT-1] += (r); /* avoid slide attacks */

// On GPU, no endian swap is needed
// Make this a nop to speed up things
#define Skein_Swap64_GPU(w64)  (w64)

// Fast versions of *LSB_First can be used on GPU
#define Skein_Put64_LSB_First_GPU(dst08,src64,bCnt) memcpy(dst08,src64,bCnt)
#define Skein_Get64_LSB_First_GPU(dst64,src08,wCnt) memcpy(dst64,src08,8*(wCnt))

// Use fast version of process_block for GPU
#define BLK_BITS        (WCNT*64)
#define KW_TWK_BASE     (0)
#define KW_KEY_BASE     (3)
#define ks              (kw + KW_KEY_BASE)                
#define ts              (kw + KW_TWK_BASE)

/*****************************************
 * Fast version of process block for GPU
 *****************************************/
__device__ void Skein_512_Process_Block_GPU(Skein_512_Ctxt_t *ctx,const u08b_t *blkPtr,size_t blkCnt,size_t byteCntAdd) {
    enum { WCNT = SKEIN_512_STATE_WORDS };
#undef  RCNT
#define RCNT  (SKEIN_512_ROUNDS_TOTAL/8)

#define SKEIN_UNROLL_512 (0)

    u64b_t  kw[WCNT+4];                         /* key schedule words : chaining vars + tweak */

    u64b_t  X0,X1,X2,X3,X4,X5,X6,X7;            /* local copy of vars, for speed */
    u64b_t  w [WCNT];                           /* local copy of input block */

    ts[0] = ctx->h.T[0];
    ts[1] = ctx->h.T[1];
    do  {
        /* this implementation only supports 2**64 input bytes (no carry out here) */
        ts[0] += byteCntAdd;                    /* update processed length */

        /* precompute the key schedule for this block */
        ks[0] = ctx->X[0];
        ks[1] = ctx->X[1];
        ks[2] = ctx->X[2];
        ks[3] = ctx->X[3];
        ks[4] = ctx->X[4];
        ks[5] = ctx->X[5];
        ks[6] = ctx->X[6];
        ks[7] = ctx->X[7];
        ks[8] = ks[0] ^ ks[1] ^ ks[2] ^ ks[3] ^ 
                ks[4] ^ ks[5] ^ ks[6] ^ ks[7] ^ SKEIN_KS_PARITY;

        ts[2] = ts[0] ^ ts[1];

        Skein_Get64_LSB_First_GPU(w,blkPtr,WCNT); /* get input block in little-endian format */

        X0   = w[0] + ks[0];                    /* do the first full key injection */
        X1   = w[1] + ks[1];
        X2   = w[2] + ks[2];
        X3   = w[3] + ks[3];
        X4   = w[4] + ks[4];
        X5   = w[5] + ks[5] + ts[0];
        X6   = w[6] + ks[6] + ts[1];
        X7   = w[7] + ks[7];

        blkPtr += SKEIN_512_BLOCK_BYTES;

#define Round512(p0,p1,p2,p3,p4,p5,p6,p7,ROT,rNum)                  \
		X##p0 += X##p1; X##p1 = RotL_64_GPU(X##p1,ROT##_0); X##p1 ^= X##p0; \
		X##p2 += X##p3; X##p3 = RotL_64_GPU(X##p3,ROT##_1); X##p3 ^= X##p2; \
		X##p4 += X##p5; X##p5 = RotL_64_GPU(X##p5,ROT##_2); X##p5 ^= X##p4; \
		X##p6 += X##p7; X##p7 = RotL_64_GPU(X##p7,ROT##_3); X##p7 ^= X##p6; \
                      
#define R512(p0,p1,p2,p3,p4,p5,p6,p7,ROT,rNum)      /* unrolled */  \
		Round512(p0,p1,p2,p3,p4,p5,p6,p7,ROT,rNum)                      \

#define I512(R)                                                     \
		X0   += ks[((R)+1) % 9];   /* inject the key schedule value */  \
		X1   += ks[((R)+2) % 9];                                        \
		X2   += ks[((R)+3) % 9];                                        \
		X3   += ks[((R)+4) % 9];                                        \
		X4   += ks[((R)+5) % 9];                                        \
		X5   += ks[((R)+6) % 9] + ts[((R)+1) % 3];                      \
		X6   += ks[((R)+7) % 9] + ts[((R)+2) % 3];                      \
		X7   += ks[((R)+8) % 9] +     (R)+1;                            \

#define R512_8_rounds(R)  /* do 8 full rounds */  \
        R512(0,1,2,3,4,5,6,7,R_512_0,8*(R)+ 1);   \
        R512(2,1,4,7,6,5,0,3,R_512_1,8*(R)+ 2);   \
        R512(4,1,6,3,0,5,2,7,R_512_2,8*(R)+ 3);   \
        R512(6,1,0,7,2,5,4,3,R_512_3,8*(R)+ 4);   \
        I512(2*(R));                              \
        R512(0,1,2,3,4,5,6,7,R_512_4,8*(R)+ 5);   \
        R512(2,1,4,7,6,5,0,3,R_512_5,8*(R)+ 6);   \
        R512(4,1,6,3,0,5,2,7,R_512_6,8*(R)+ 7);   \
        R512(6,1,0,7,2,5,4,3,R_512_7,8*(R)+ 8);   \
        I512(2*(R)+1);        /* and key injection */

        R512_8_rounds( 0);

#define R512_Unroll_R(NN) ((SKEIN_UNROLL_512 == 0 && SKEIN_512_ROUNDS_TOTAL/8 > (NN)) || (SKEIN_UNROLL_512 > (NN)))

        R512_8_rounds( 1);
        R512_8_rounds( 2);
        R512_8_rounds( 3);
        R512_8_rounds( 4);
        R512_8_rounds( 5);
        R512_8_rounds( 6);
        R512_8_rounds( 7);
        R512_8_rounds( 8);

        /* do the final "feedforward" xor, update context chaining vars */
        ctx->X[0] = X0 ^ w[0];
        ctx->X[1] = X1 ^ w[1];
        ctx->X[2] = X2 ^ w[2];
        ctx->X[3] = X3 ^ w[3];
        ctx->X[4] = X4 ^ w[4];
        ctx->X[5] = X5 ^ w[5];
        ctx->X[6] = X6 ^ w[6];
        ctx->X[7] = X7 ^ w[7];

        ts[1] &= ~SKEIN_T1_FLAG_FIRST;

    }  while (--blkCnt);
    
	ctx->h.T[0] = ts[0];
    ctx->h.T[1] = ts[1];

} // Skein_512_Process_Block_GPU


 /*******************************
 * Process the input bytes 
 ********************************/
__device__ int Skein_512_Update_GPU(Skein_512_Ctxt_t *ctx, const u08b_t *msg, size_t msgByteCnt) {
    size_t n;

   // Skein_Assert(ctx->h.bCnt <= SKEIN_512_BLOCK_BYTES,SKEIN_FAIL);     /* catch uninitialized context */

    /* process full blocks, if any */
    if (msgByteCnt + ctx->h.bCnt > SKEIN_512_BLOCK_BYTES) {
		 /* finish up any buffered message data */
        if (ctx->h.bCnt) {
            n = SKEIN_512_BLOCK_BYTES - ctx->h.bCnt;  /* # bytes free in buffer b[] */
            if (n) {
               // Skein_assert(n < msgByteCnt);         /* check on our logic here */
                memcpy(&ctx->b[ctx->h.bCnt],msg,n);
                msgByteCnt  -= n;
                msg         += n;
                ctx->h.bCnt += n;
            }
           // Skein_assert(ctx->h.bCnt == SKEIN_512_BLOCK_BYTES);
            Skein_512_Process_Block_GPU(ctx,ctx->b,1,SKEIN_512_BLOCK_BYTES);
            ctx->h.bCnt = 0;
        }
        /* now process any remaining full blocks, directly from input message data */
        if (msgByteCnt > SKEIN_512_BLOCK_BYTES) {
            n = (msgByteCnt-1) / SKEIN_512_BLOCK_BYTES;   /* number of full blocks to process */
            Skein_512_Process_Block_GPU(ctx,msg,n,SKEIN_512_BLOCK_BYTES);
            msgByteCnt -= n * SKEIN_512_BLOCK_BYTES;
            msg        += n * SKEIN_512_BLOCK_BYTES;
        }
       // Skein_assert(ctx->h.bCnt == 0);
    }

    /* copy any remaining source message data bytes into b[] */
    if (msgByteCnt) {
       // Skein_assert(msgByteCnt + ctx->h.bCnt <= SKEIN_512_BLOCK_BYTES);
        memcpy(&ctx->b[ctx->h.bCnt],msg,msgByteCnt);
        ctx->h.bCnt += msgByteCnt;
    }

    return SKEIN_SUCCESS;
}

/********************************
 * Process data to be hashed
 ********************************/
__device__ int Skein_Update_GPU(hashState *state, const BitSequence *data, DataLength databitlen) {

	if ((databitlen & 7) == 0) {
		return Skein_512_Update_GPU(&state->u.ctx_512, data, databitlen >> 3);
	}
	else {
		size_t bCnt = (databitlen >> 3) + 1;                  /* number of bytes to handle */
		u08b_t mask, *p;

		Skein_512_Update_GPU(&state->u.ctx_512, data,bCnt);
		p    = state->u.ctx_512.b;

		Skein_Set_Bit_Pad_Flag(state->u.h);                     /* set tweak flag for the final call */
		/* now "pad" the final partial byte the way NIST likes */
		bCnt = state->u.h.bCnt;         /* get the bCnt value (same location for all block sizes) */
		//Skein_assert(bCnt != 0);        /* internal sanity check: there IS a partial byte in the buffer! */
		mask = (u08b_t) (1u << (7 - (databitlen & 7)));         /* partial byte bit mask */
		p[bCnt-1]  = (u08b_t)((p[bCnt-1] & (0-mask)) | mask);   /* apply bit padding on final byte (in the buffer) */

		return SUCCESS;
	}
}

/****************************************************************
 * Finalize the hash computation and output the block, no OUTPUT stage
 ***************************************************************/
__device__ int Skein_512_Final_Pad_GPU(Skein_512_Ctxt_t *ctx, u08b_t *hashVal) {
    //Skein_Assert(ctx->h.bCnt <= SKEIN_512_BLOCK_BYTES,SKEIN_FAIL);    /* catch uninitialized context */

    ctx->h.T[1] |= SKEIN_T1_FLAG_FINAL;        /* tag as the final block */
    if (ctx->h.bCnt < SKEIN_512_BLOCK_BYTES)   /* zero pad b[] if necessary */
        memset(&ctx->b[ctx->h.bCnt],0,SKEIN_512_BLOCK_BYTES - ctx->h.bCnt);
    Skein_512_Process_Block_GPU(ctx,ctx->b,1,ctx->h.bCnt);    /* process the final block */
    
    Skein_Put64_LSB_First_GPU(hashVal,ctx->X,SKEIN_512_BLOCK_BYTES);   /* "output" the state bytes */
    
    return SKEIN_SUCCESS;
}

/****************************************************************
 * Init the context for a MAC and/or tree hash operation
 ***************************************************************/
__device__ int Skein_512_InitExt_GPU(Skein_512_Ctxt_t *ctx,size_t hashBitLen,u64b_t treeInfo, const u08b_t *key, size_t keyBytes) {
    union {
        u08b_t  b[SKEIN_512_STATE_BYTES];
        u64b_t  w[SKEIN_512_STATE_WORDS];
    } cfg;                              /* config block */
        
    //Skein_Assert(hashBitLen > 0,SKEIN_BAD_HASHLEN);
    //Skein_Assert(keyBytes == 0 || key != NULL,SKEIN_FAIL);

    /* compute the initial chaining values ctx->X[], based on key */
	/* is there a key? */
    if (keyBytes == 0) {
		/* no key: use all zeroes as key for config block */
        memset(ctx->X,0,sizeof(ctx->X));
    } else {
		 /* here to pre-process a key */
        //Skein_assert(sizeof(cfg.b) >= sizeof(ctx->X));
        /* do a mini-Init right here */
        ctx->h.hashBitLen=8*sizeof(ctx->X);     /* set output hash bit count = state size */
        Skein_Start_New_Type(ctx,KEY);          /* set tweaks: T0 = 0; T1 = KEY type */
        memset(ctx->X,0,sizeof(ctx->X));        /* zero the initial chaining variables */
        Skein_512_Update_GPU(ctx,key,keyBytes);     /* hash the key */
        Skein_512_Final_Pad_GPU(ctx,cfg.b);         /* put result into cfg.b[] */
        memcpy(ctx->X,cfg.b,sizeof(cfg.b));     /* copy over into ctx->X[] */
    }
    /* build/process the config block, type == CONFIG (could be precomputed for each key) */
    ctx->h.hashBitLen = hashBitLen;             /* output hash bit count */
    Skein_Start_New_Type(ctx,CFG_FINAL);

    memset(&cfg.w,0,sizeof(cfg.w));             /* pre-pad cfg.w[] with zeroes */
    cfg.w[0] = Skein_Swap64_GPU(SKEIN_SCHEMA_VER);
    cfg.w[1] = Skein_Swap64_GPU(hashBitLen);        /* hash result length in bits */
    cfg.w[2] = Skein_Swap64_GPU(treeInfo);          /* tree hash config info (or SKEIN_CFG_TREE_INFO_SEQUENTIAL) */

    //Skein_Show_Key(512,&ctx->h,key,keyBytes);

    /* compute the initial chaining values from config block */
    Skein_512_Process_Block_GPU(ctx,cfg.b,1,SKEIN_CFG_STR_LEN);

    /* The chaining vars ctx->X are now initialized */
    /* Set up to process the data message portion of the hash (default) */
    ctx->h.bCnt = 0;                            /* buffer b[] starts out empty */
    Skein_Start_New_Type(ctx,MSG);
    
    return SKEIN_SUCCESS;
}


/***********************************************************************************
 * Kernel to compute a single UBI block of variable size
 ***********************************************************************************/
__global__ void SkeinTree_UBI_Kernel( u08b_t *Md, hashState *sd, uint_t height, uint_t node, 
	uint_t leaf, size_t bCnt, uint_t blkBytes, size_t hashBits, u64b_t treeInfo ) { 
	 
	//int tx = threadIdx.x; // Thread index
	//int bx = blockIdx.x;  // Block index
	int threadID = blockIdx.x * TILE_SIZE + threadIdx.x; // Global thread ID

	uint_t nodeLen = blkBytes << ((height) ? node : leaf);
	uint_t srcOffs = nodeLen * threadID;
	uint_t dstOffs = blkBytes * threadID;
	
	if( srcOffs < bCnt ) {
		//printf( "thread %d: blkBytes = %d, nodeLen = %d, bCnt = %d, srcOffs = %d, dstOffs = %d\n", threadID, blkBytes, nodeLen, bCnt, srcOffs, dstOffs );
		Skein_512_InitExt_GPU(&sd[threadID].u.ctx_512, (size_t)hashBits, treeInfo, NULL, 0);

		uint_t n = bCnt - srcOffs;         /* number of bytes left at this level */
		/* limit to node size */
		if (n > nodeLen)
			n = nodeLen;
		sd[threadID].u.h.T[0] = srcOffs;       /* nonzero initial offset in tweak! */
		Skein_Set_Tree_Level(sd[threadID].u.h, height+1);

		Skein_Update_GPU(&sd[threadID], Md + srcOffs, n*8);

		 /* finish up this node, output intermediate result to M[]*/
		Skein_512_Final_Pad_GPU(&sd[threadID].u.ctx_512, Md + dstOffs);
	}

} // SkeinTree_UBI_Kernel

/*********************************************************
 * Skein tree hashing on the GPU
 * Sets up and launches kernels for each tree level
 *********************************************************/
int SkeinTreeHash_GPU( uint_t blkSize, uint_t hashBits, const u08b_t *msg, size_t msgBytes,
	uint_t leaf, uint_t node, uint_t maxLevel, u08b_t *hashRes, uint_t skein_DebugFlag ) {

	enum      { MAX_HEIGHT = 32 }; /* how deep we can go here */
	uint_t    height;
	uint_t    blkBytes  = blkSize/8;
	uint_t    saveDebug = skein_DebugFlag;
	size_t    nodeLen, bCnt;
	u64b_t    treeInfo;
	u08b_t    *M;
	//u08b_t    M[MAX_TREE_MSG_LEN+4];
	hashState G, s;

	//cudaMallocHost( (void**) &M, (MAX_TREE_MSG_LEN+4) * sizeof(u08b_t) );
	// Use pinned memory for speed
	//printf( "before CUDA malloc host\n");
	cudaMallocHost( (void**) &M, msgBytes + 4 );
	//printf( "after CUDA malloc host\n");

	// CUDA
	cudaError_t error;
	u08b_t *Md;

	/* precompute the config block result G for multiple uses below */
#ifdef SKEIN_DEBUG
	if (skein_DebugFlag)
		skein_DebugFlag |= SKEIN_DEBUG_CONFIG;
#endif
	treeInfo = SKEIN_CFG_TREE_INFO(leaf, node, maxLevel);
	if (Skein_512_InitExt(&G.u.ctx_512, (size_t)hashBits, treeInfo, NULL, 0) != SKEIN_SUCCESS )
		FatalError("Skein_512_InitExt() fails in GPU tree");
	skein_DebugFlag = saveDebug;

	bCnt = msgBytes;
	memcpy(M, msg, bCnt);

	// Allocate memory on device for message
	//cudaMalloc( (void**) &Md, (MAX_TREE_MSG_LEN+4) * sizeof(u08b_t) );
	//printf( "before CUDA malloc\n");
	cudaMalloc( (void**) &Md, msgBytes + 4 );
	//printf( "after CUDA malloc\n");
	//cudaMemcpy( Md, M, (MAX_TREE_MSG_LEN+4) * sizeof(u08b_t), cudaMemcpyHostToDevice );
	cudaMemcpy( Md, M, msgBytes + 4, cudaMemcpyHostToDevice );
	//printf( "after CUDA memcpy\n");

	// Walk up the tree
	for (height = 0; ; height++) {

		nodeLen = blkBytes << ((height) ? node : leaf);
		uint_t numNodes = (bCnt / nodeLen) + ((bCnt % nodeLen) ? 1 : 0);
		//printf( "GPU: height = %d, bCnt = %d, numNodes = %d, nodeLen = %d\n", height, bCnt, numNodes, nodeLen );

		// Done yet (only one block left)?
		if (height && (bCnt==blkBytes)) {
			//printf( "GPU: hit last block\n");
			break;
		}

		// Final allowed level?
		if (height+1 == maxLevel ) {
			// If so, do it as one big hash
			//printf( "GPU: hit last level\n");
			s = G;
			// Copy in updated message for this case
			cudaMemcpy( M, Md, msgBytes, cudaMemcpyDeviceToHost );
			Skein_Set_Tree_Level(s.u.h, height+1);
			Skein_Update   (&s, M, bCnt*8);
			Skein_512_Final_Pad(&s.u.ctx_512, M);
			break;
		}

		// 
		// Lower levels of tree are parallelized
		//
		hashState *sd;
		cudaMalloc( (void**) &sd, numNodes * sizeof(G) );

		// Organize stuff for GPU
		dim3 dimBlock(TILE_SIZE, 1);
		dim3 dimGrid((int)ceil((float)numNodes / (float)TILE_SIZE), 1);
		//dim3 dimGrid((int)ceil((float)(msgBytes) / (float)blkBytes), 1);
		//printf( "dimGrid = (%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z );

		SkeinTree_UBI_Kernel<<< dimGrid, dimBlock >>>( Md, sd, height, node, leaf, bCnt, blkBytes, (size_t)hashBits, treeInfo );
		cudaThreadSynchronize();

		bCnt = blkBytes * (bCnt / nodeLen) + ((bCnt % nodeLen) ? blkBytes : 0);

		// Copy state back to host only on last iteration
		if (height+1 && (bCnt==blkBytes)) {
		//	printf( "GPU: copying state back\n" );
			cudaMemcpy( &s, &sd[0], sizeof(sd[0]), cudaMemcpyDeviceToHost );
		}
		cudaFree(sd);

	}

	// Output UBI
	Skein_512_Output(&s.u.ctx_512, hashRes);

	// Cleanup
	cudaFreeHost(M);

	// Make sure everything worked okay; if not, indicate that error occurred
	error = cudaGetLastError();
	if(error) {
		printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
		return 1;
	}

	return 0;

}