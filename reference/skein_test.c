/***********************************************************************
 **
 ** Test/verification code for the Skein block functions.
 **
 ** Source code author: 
 **	Doug Whiting, 2008: Original reference implementation
 **	Matt Kelly & Tom Connors, 2013: Parallelized tree hashing using CUDA
 **
 ** This algorithm and source code is released to the public domain.
 **
 ** Testing:
 **   - buffering of incremental calls (random cnt steps)
 **   - partial input byte handling
 **   - output sample hash results (for comparison of ref vs. optimized)
 **   - performance
 **
 ***********************************************************************/

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <assert.h>

#include "skein.h"
#include "SHA3api_ref.h"

static const uint_t HASH_BITS[] =    /* list of hash hash lengths to test */
{ 160,224,256,384,512,1024, 256+8,512+8,1024+8,2048+8 };

#define HASH_BITS_CNT   (sizeof(HASH_BITS)/sizeof(HASH_BITS[0]))

/* bits of the verbose flag word */
#define V_KAT_LONG      (1u << 0)
#define V_KAT_SHORT     (1u << 1)
#define V_KAT_NO_TREE   (1u << 2)
#define V_KAT_NO_SEQ    (1u << 3)
#define V_KAT_NO_3FISH  (1u << 4)
#define V_KAT_DO_3FISH  (1u << 5)

/* External function to process blkCnt (nonzero) full block(s) of data. */
void Skein_512_Process_Block(Skein_512_Ctxt_t *ctx, const u08b_t *blkPtr, size_t blkCnt, size_t byteCntAdd);

/********************** debug i/o helper routines **********************/

/* print out a msg and exit with an error code */
void FatalError(const char *s,...) { 
	va_list ap;
	va_start(ap,s);
	vprintf(s,ap);
	va_end(ap);
	printf("\n");
	exit(2);
} // FatalError

static uint_t _quiet_   =   0;  /* quiet processing? */
static uint_t verbose   =   0;  /* verbose flag bits */

/* formatted output of byte array */
void ShowBytes(uint_t cnt,const u08b_t *b) {
	uint_t i;
	for (i=0;i < cnt;i++) {
		if (i % 16 ==  0) printf("    ");
		else if (i % 4 == 0) printf(" ");
		printf(" %02X",b[i]);
		if (i %16 == 15 || i==cnt-1) printf("\n");
	}
} // ShowBytes

#ifndef SKEIN_DEBUG
uint_t skein_DebugFlag     =   0;     /* dummy flags (if not defined elsewhere) */
#endif

#define SKEIN_DEBUG_SHORT   (SKEIN_DEBUG_HDR | SKEIN_DEBUG_STATE | SKEIN_DEBUG_TWEAK | SKEIN_DEBUG_KEY | SKEIN_DEBUG_INPUT_08 | SKEIN_DEBUG_FINAL)
#define SKEIN_DEBUG_DEFAULT (SKEIN_DEBUG_SHORT)

void Show_Debug(const char *s,...) {
	/* are we showing debug info? */
	if (skein_DebugFlag) {
		va_list ap;
		va_start(ap,s);
		vprintf(s,ap);
		va_end(ap);
	}
} // Show_Debug

/********************** use RC4 to generate test data ******************/
/* Note: this works identically on all platforms (big/little-endian)   */
static struct {
	uint_t I,J;                         /* RC4 vars */
	u08b_t state[256];
} prng;

void RandBytes(void *dst,uint_t byteCnt) {
	u08b_t a,b;
	u08b_t *d = (u08b_t *) dst;
	/* run RC4  */
	for (;byteCnt;byteCnt--,d++) {
		prng.I  = (prng.I+1) & 0xFF;
		a       =  prng.state[prng.I];
		prng.J  = (prng.J+a) & 0xFF;
		b       =  prng.state[prng.J];
		prng.state[prng.I] = b;
		prng.state[prng.J] = a;
		*d      =  prng.state[(a+b) & 0xFF];
	}
} // RandBytes

/* get a pseudo-random 32-bit integer in a portable way */
uint_t Rand32(void) {
	uint_t i,n;
	u08b_t tmp[4];

	RandBytes(tmp,sizeof(tmp));

	for (i=n=0;i<sizeof(tmp);i++)
		n = n*256 + tmp[i];

	return n;
} // Rand32

/* init the (RC4-based) prng */
void Rand_Init(u64b_t seed) {
	uint_t i,j;
	u08b_t tmp[512];

	/* init the "key" in an endian-independent fashion */
	for (i=0;i<8;i++)
		tmp[i] = (u08b_t) (seed >> (8*i));

	/* initialize the permutation */
	for (i=0;i<256;i++)
		prng.state[i]=(u08b_t) i;

	/* now run the RC4 key schedule */
	for (i=j=0;i<256;i++)
	{                   
		j = (j + prng.state[i] + tmp[i%8]) & 0xFF;
		tmp[256]      = prng.state[i];
		prng.state[i] = prng.state[j];
		prng.state[j] = tmp[256];
	}
	prng.I = prng.J = 0;  /* init I,J variables for RC4 */

	/* discard initial keystream before returning */
	RandBytes(tmp,sizeof(tmp));
} // Rand_Init


/***********************************************************************/
/* An AHS-like API that allows explicit setting of block size          */
/*    [i.e., the AHS API selects a block size based solely on the ]    */
/*    [hash result length, while Skein allows independent hash    ]    */
/*    [result size and block size                                 ]    */
/***********************************************************************/

/* process data to be hashed */
int Skein_Update(hashState *state, const BitSequence *data, DataLength databitlen) {
	/* only the final Update() call is allowed do partial bytes, else assert an error */
	Skein_Assert((state->u.h.T[1] & SKEIN_T1_FLAG_BIT_PAD) == 0 || databitlen == 0, FAIL);

	if ((databitlen & 7) == 0) {
		return Skein_512_Update(&state->u.ctx_512,data,databitlen >> 3);
	}
	else {
		size_t bCnt = (databitlen >> 3) + 1;                  /* number of bytes to handle */
		u08b_t mask,*p;

#if (!defined(_MSC_VER)) || (MSC_VER >= 1200)                 /* MSC v4.2 gives (invalid) warning here!!  */
		/* sanity checks: allow u.h --> all contexts */
		Skein_assert(&state->u.h == &state->u.ctx_512.h);
#endif
		Skein_512_Update(&state->u.ctx_512,data,bCnt);
		p    = state->u.ctx_512.b;

		Skein_Set_Bit_Pad_Flag(state->u.h);                     /* set tweak flag for the final call */
		/* now "pad" the final partial byte the way NIST likes */
		bCnt = state->u.h.bCnt;         /* get the bCnt value (same location for all block sizes) */
		Skein_assert(bCnt != 0);        /* internal sanity check: there IS a partial byte in the buffer! */
		mask = (u08b_t) (1u << (7 - (databitlen & 7)));         /* partial byte bit mask */
		p[bCnt-1]  = (u08b_t)((p[bCnt-1] & (0-mask)) | mask);   /* apply bit padding on final byte (in the buffer) */

		return SUCCESS;
	}
}

/* filter out <blkSize,hashBits> pairs in short KAT mode */
uint_t Short_KAT_OK(uint_t blkSize,uint_t hashBits) {
	if (hashBits != 256 && hashBits != 384 && hashBits != 512)
		return 0;
	return 1;
} // Short_KAT_OK

#define MAX_TREE_MSG_LEN  (1 << 12)

/* generate a KAT test for the given data and tree parameters. */
/* This is an "all-in-one" call. It is not intended to represent */
/* how a real multi-processor version would be implemented, but  */
/* the results will be the same */
void Skein_TreeHash
	(uint_t blkSize,uint_t hashBits,const u08b_t *msg,size_t msgBytes,
	uint_t leaf   ,uint_t node    ,uint_t maxLevel  ,u08b_t *hashRes) {

	enum      { MAX_HEIGHT = 32 };          /* how deep we can go here */
	uint_t    height;
	uint_t    blkBytes  = blkSize/8;
	uint_t    saveDebug = skein_DebugFlag;
	size_t    n,nodeLen,srcOffs,dstOffs,bCnt;
	u64b_t    treeInfo;
	u08b_t    M[MAX_TREE_MSG_LEN+4];
	hashState G,s;

	assert(node < 256 && leaf < 256 && maxLevel < 256);
	assert(node >  0  && leaf >  0  && maxLevel >  1 );
	assert(blkSize == 256 || blkSize == 512 || blkSize == 1024);
	assert(blkBytes <= sizeof(M));
	assert(msgBytes <= sizeof(M));

	/* precompute the config block result G for multiple uses below */
#ifdef SKEIN_DEBUG
	if (skein_DebugFlag)
		skein_DebugFlag |= SKEIN_DEBUG_CONFIG;
#endif
	treeInfo = SKEIN_CFG_TREE_INFO(leaf,node,maxLevel);
	if (Skein_512_InitExt(&G.u.ctx_512,(size_t) hashBits,treeInfo,NULL,0) != SKEIN_SUCCESS )
		FatalError("Skein_512_InitExt() fails in tree");
	skein_DebugFlag = saveDebug;

	bCnt = msgBytes;
	memcpy(M,msg,bCnt);

	/* walk up the tree */
	for (height=0;;height++) {

		/* are we done (with only one block left)? */
		if (height && (bCnt==blkBytes))
			break;

		/* is this the final allowed level? */
		if (height+1 == maxLevel) {
			/* if so, do it as one big hash */
			s = G;
			Skein_Set_Tree_Level(s.u.h,height+1);
			Skein_Update   (&s,M,bCnt*8);
			Skein_512_Final_Pad(&s.u.ctx_512,M);
			break;
		}

		nodeLen = blkBytes << ((height) ? node : leaf);
		for (srcOffs=dstOffs=0;srcOffs <= bCnt;) {
			n = bCnt - srcOffs;         /* number of bytes left at this level */
			if (n > nodeLen)            /* limit to node size */
				n = nodeLen;
			s = G;
			s.u.h.T[0] = srcOffs;       /* nonzero initial offset in tweak! */
			Skein_Set_Tree_Level(s.u.h,height+1);
			Skein_Update   (&s,M+srcOffs,n*8);
			Skein_512_Final_Pad(&s.u.ctx_512,M+dstOffs);  /* finish up this node, output intermediate result to M[]*/
			dstOffs+=blkBytes;
			srcOffs+=n;
			if (srcOffs >= bCnt)        /* special logic to handle (msgBytes == 0) case */
				break;
		}
		bCnt = dstOffs;

	} // walk tree

	/* output the result */
	Skein_512_Output(&s.u.ctx_512, hashRes);

} // Skein_TreeHash

/*
** Generate tree-mode hash KAT vectors.
** Note:
**    Tree vectors are different enough from non-tree vectors that it 
**    makes sense to separate this out into a different function, rather 
**    than shoehorn it into the same KAT logic as the other modes.
**/
void Skein_GenKAT_Tree(uint_t blkSize) {

	static const struct {
		uint_t leaf,node,maxLevel,levels;
	}

	TREE_PARMS[] = { {2,2,2,2}, {1,2,3,2}, {2,1,0xFF,3} };
#define TREE_PARM_CNT (sizeof(TREE_PARMS)/sizeof(TREE_PARMS[0]))

	u08b_t  msg[MAX_TREE_MSG_LEN+4],hashVal[MAX_TREE_MSG_LEN+4];
	uint_t  i,j,k,n,p,q,hashBits,node,leaf,leafBytes,msgBytes,byteCnt,levels,maxLevel;

	assert(blkSize == 256 || blkSize == 512 || blkSize == 1024);
	for (i=0;i<MAX_TREE_MSG_LEN;i+=2) {   
		/* generate "incrementing" tree hash input msg data */
		msg[i  ] = (u08b_t) ((i ^ blkSize) ^ (i >> 16));
		msg[i+1] = (u08b_t) ((i ^ blkSize) >> 8);
	}

	for (k=q=n=0;k < HASH_BITS_CNT;k++) {
		hashBits = HASH_BITS[k];
		if (!Short_KAT_OK(blkSize,hashBits))
			continue;
		if ((verbose & V_KAT_SHORT) && (hashBits != blkSize))
			continue;
		for (p=0;p <TREE_PARM_CNT;p++) {
			if (p && (verbose & V_KAT_SHORT))
				continue;           /* keep short KATs short */
			if (p && hashBits != blkSize)
				continue;           /* we only need one "non-full" size */

			leaf      = TREE_PARMS[p].leaf;
			node      = TREE_PARMS[p].node;
			maxLevel  = TREE_PARMS[p].maxLevel;
			levels    = TREE_PARMS[p].levels;
			leafBytes = (blkSize/8) << leaf;    /* number of bytes in a "full" leaf */

			/* different numbers of leaf results */
			for (j=0;j<4;j++) {
				if ((verbose & V_KAT_SHORT) && (j != 3) && (j != 0))
					continue;
				if (j && (hashBits != blkSize))
					break;
				switch (j) {
				case 0: 
					n = 1; 
					break;
				case 1: 
					n = 2; 
					break;         
				case 2: 
					n = (1 << (node * (levels-2)))*3/2;
					if (n <= 2) continue; 
					break;
				case 3: 
					n = (1 << (node * (levels-1)));
					break;
				}
				byteCnt = n*leafBytes;
				assert(byteCnt > 0);
				if (byteCnt > MAX_TREE_MSG_LEN)
					continue;
				q = (q+1) % leafBytes;
				msgBytes = byteCnt - q;

				printf("\n:Skein-512: %4d-bit hash, msgLen =%6d bits",hashBits,msgBytes*8);
				printf(". Tree: leaf=%02X, node=%02X, maxLevels=%02X\n",leaf,node,maxLevel);
				printf("\nMessage data:\n");
				if (msgBytes == 0)
					printf("    (none)\n");
				else
					ShowBytes(msgBytes,msg);

				Skein_TreeHash(blkSize,hashBits,msg,msgBytes,leaf,node,maxLevel,hashVal);

				printf("Result:\n");
				ShowBytes((hashBits+7)/8,hashVal);
				printf("--------------------------------\n");

			}
		}
	}

} // Skein_GetKAT_Tree

/* Print usage */
void GiveHelp(void) {
	printf("Syntax:  skein_test [options]\n"
		"Options: -bNN  = set Skein block size to NN bits\n"
		"         -lNN  = set max test length  to NN bits\n"
		"         -tNN  = set Skein hash length to NN bits\n"
		"         -sNN  = set initial random seed\n"
		"         -k    = output KAT results to stdout\n"
		);
	exit(2);
} // GiveHelp

/* Main function */
int main(int argc,char *argv[]) {

	int    i;
	uint_t doKAT   =    0;   /* generate KAT vectors?    */
	uint_t blkSize =    0;   /* Skein state size in bits */
	uint_t maxLen  = 1024;   /* max block size   in bits */
	uint_t hashLen =    0;   /* hash length      in bits (0 --> all) */
	uint_t seed0   = (uint_t) time(NULL); /* randomize based on time */
	uint_t oneBlk  =    0;   /* test block size */

	 /* process command-line switches */
	for (i=1;i<argc;i++) {
		if (argv[i][0] == '-') {
			switch(toupper(argv[i][1])) {
				case '?': GiveHelp();                         break;
				case 'B': blkSize       |= atoi(argv[i]+2);   break;
				case 'L': maxLen         = atoi(argv[i]+2);   break;
				case 'S': seed0          = atoi(argv[i]+2);   break;
				case 'T': hashLen        = atoi(argv[i]+2);   break;
				case 'K': doKAT          = 1;                 break;
				case 'V': verbose       |= (argv[i][2]) ? atoi(argv[i]+2) : V_KAT_LONG; break;
				case 'D': 
					switch (toupper(argv[i][2])) {

						#ifdef SKEIN_DEBUG
						case  0 : skein_DebugFlag |= SKEIN_DEBUG_DEFAULT; break;
						case '-': skein_DebugFlag |= SKEIN_DEBUG_SHORT;   break;
						case '+': skein_DebugFlag |= SKEIN_DEBUG_ALL;     break;
						case 'P': skein_DebugFlag |= SKEIN_DEBUG_PERMUTE; break;
						case 'I': skein_DebugFlag |= SKEIN_DEBUG_SHORT |  SKEIN_DEBUG_INJECT; break;
						case 'C': skein_DebugFlag |= SKEIN_DEBUG_SHORT & ~SKEIN_DEBUG_CONFIG; break;
						#endif

				default : skein_DebugFlag |= atoi(argv[i]+2);     break;
					}
					break;
				
				default:  FatalError("Unsupported command-line option: %s",argv[i]);
					break;
			}
		}
		else if (argv[i][0] == '?')
			GiveHelp();
		else if (isdigit(argv[i][0]))
			oneBlk = atoi(argv[i]);
	}

	/* default is all block sizes */
	if (blkSize == 0)
		blkSize = 256 | 512 | 1024;

	if (doKAT) {
		Skein_GenKAT_Tree(blkSize);
	} 

	return 0;
}
