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

#include "SkeinTest.h"

#ifndef SKEIN_DEBUG
	uint_t skein_DebugFlag; /* dummy flags (if not defined elsewhere) */
#endif

static uint_t _quiet_ = 0;  /* quiet processing? */
static uint_t verbose = 0;  /* verbose flag bits */

 /* print out a msg and exit with an error code */
void FatalError(const char *s, ...) { 
	va_list ap;
	va_start(ap,s);
	vprintf(s,ap);
	va_end(ap);
	printf("\n");
	exit(2);
}

/* formatted output of byte array */
void ShowBytes(uint_t cnt,const u08b_t *b) {
	uint_t i;
	for (i=0;i < cnt;i++) {
		if (i % 16 ==  0) printf("    ");
		else if (i % 4 == 0) printf(" ");
		printf(" %02X", b[i]);
		if (i % 16 == 15 || i == cnt-1) printf("\n");
	}
}

void Show_Debug(const char *s, ...) {
	if (skein_DebugFlag) {
		va_list ap;
		va_start(ap,s);
		vprintf(s,ap);
		va_end(ap);
	}
}

/* process data to be hashed */
int Skein_Update(hashState *state, const BitSequence *data, DataLength databitlen) {

	if ((databitlen & 7) == 0) {
		return Skein_512_Update(&state->u.ctx_512, data, databitlen >> 3);
	} else {
		size_t bCnt = (databitlen >> 3) + 1;                  /* number of bytes to handle */
		u08b_t mask, *p;

		Skein_512_Update(&state->u.ctx_512, data,bCnt);
		p    = state->u.ctx_512.b;

		Skein_Set_Bit_Pad_Flag(state->u.h);                     /* set tweak flag for the final call */
		/* now "pad" the final partial byte the way NIST likes */
		bCnt = state->u.h.bCnt;         /* get the bCnt value (same location for all block sizes) */
		Skein_assert(bCnt != 0);        /* internal sanity check: there IS a partial byte in the buffer! */
		mask = (u08b_t) (1u << (7 - (databitlen & 7)));         /* partial byte bit mask */
		p[bCnt-1]  = (u08b_t)((p[bCnt-1] & (0-mask)) | mask);   /* apply bit padding on final byte (in the buffer) */

		return SUCCESS;
	}

} // Skein_Update

/* filter out <blkSize,hashBits> pairs in short KAT mode */
uint_t Short_KAT_OK(uint_t blkSize,uint_t hashBits) {
	if (hashBits != 256 && hashBits != 384 && hashBits != 512)
		return 0;
	return 1;
} // Short_KAT_OK

/*************************************************************
 * Find number of differing bytes between two hash values
 *************************************************************/
int compareHashVals( uint_t n, const u08b_t *a, const u08b_t *b ) {
	uint_t i;
	uint_t diff = 0;
	for( i =0 ; i < n; ++i ) {
		if( a[i] != b[i] ) {
			diff++;
		}
	}
	return diff;
}

/************************************************
 * Custom long tree tests to gather speedup data
 * Message sizes should be >> 1 MB
 ************************************************/
void Skein_Gen_Long_Trees(uint_t blkSize){
	clock_t startTime, endTime;
	float cpuTime = 0.0;
	float gpuTime = 0.0;
	
	printf( "\n*******************************************************\n");
	printf( "* Running custom LONG tree tests from here\n");
	printf( "*******************************************************\n\n");

	uint_t leafsize		= MY_LEAF_SIZE;
	uint_t node_fanout  = MY_NODE_FANOUT;
	uint_t max_height	= MY_MAX_HEIGHT; 
	uint_t msgBytes		= MY_MSG_SIZE;
	uint_t hashb	    = 512;
	uint_t i;
	
	/*
	u08b_t msg[MY_MSG_SIZE+4];
	u08b_t hashValCPU[MY_MSG_SIZE+4];
	u08b_t hashValGPU[MY_MSG_SIZE+4];
	*/
	// Use dynamic allocation for very large messages
	u08b_t *msg, *hashValCPU, *hashValGPU;

	msg		   = (u08b_t*) malloc( MY_MSG_SIZE + 4 );
	hashValCPU = (u08b_t*) malloc( MY_MSG_SIZE + 4 );
	hashValGPU = (u08b_t*) malloc( MY_MSG_SIZE + 4 );
	
	/* generate "incrementing" tree hash input msg data */ 
	for( i = 0; i < MY_MSG_SIZE; i += 2){
		msg[i  ] = (u08b_t) ((i ^ blkSize) ^ (i >> 16));
		msg[i+1] = (u08b_t) ((i ^ blkSize) >> 8);
	}

	printf("\n:Skein-512: %4d-bit hash, msgLen = %d bytes\n", hashb, msgBytes);
	printf("\tTree: leaf=%02X, node=%02X, maxLevels=%02X\n", leafsize, node_fanout, max_height );
	printf("\tRunning for %d iterations (excluding warmup pass)\n", ITERATIONS);

	// Untimed warmup pass for CPU
	SkeinTreeHash_CPU(blkSize, hashb, msg, msgBytes, leafsize, node_fanout, max_height, hashValCPU, skein_DebugFlag);

	startTime = clock();
	for( int it = 0; it < ITERATIONS; it++ ) {
		SkeinTreeHash_CPU(blkSize, hashb, msg, msgBytes, leafsize, node_fanout, max_height, hashValCPU, skein_DebugFlag);
	}
	endTime = clock();
	cpuTime = (float)(endTime - startTime) * 1000 / (float)CLOCKS_PER_SEC / ITERATIONS;

	printf("\nCPU Result:\n");
	ShowBytes((hashb+7)/8,hashValCPU);
	printf( "\n\tHost computation took %.3f ms\n\n", cpuTime );
	printf("\t--------------------------------\n");

	// Untimed warmup pass for GPU
	SkeinTreeHash_GPU(blkSize, hashb, msg, msgBytes, leafsize, node_fanout, max_height, hashValGPU, skein_DebugFlag);

	startTime = clock();
	for( int it = 0; it < ITERATIONS; it++ ) {
		SkeinTreeHash_GPU(blkSize, hashb, msg, msgBytes, leafsize, node_fanout, max_height, hashValGPU, skein_DebugFlag);
	}
	endTime = clock();
	gpuTime = (float)(endTime - startTime) * 1000 / (float)CLOCKS_PER_SEC / ITERATIONS;

	printf("\nGPU Result:\n");
	ShowBytes((hashb+7)/8,hashValGPU);
	uint_t diff = compareHashVals((hashb+7)/8, hashValCPU, hashValGPU);
	if( diff ) {
		printf( "\nERROR: %d different bytes between host and device results!\n", diff);
	}
	printf( "\n\tDevice computation took %.3f ms\n\n", gpuTime );
	printf("*********************************************************\n");

	// Cleanup
	free(msg);
	free(hashValCPU);
	free(hashValGPU);

}


/*********************************************************************************
 * Generate tree-mode hash KAT vectors.
 * These test vectors are useful for checking correctness against NIST submission
 * They are not very useful for finding speedup due to small message size.
 *********************************************************************************/

void Skein_GenKAT_Tree(uint_t blkSize) {

	static const struct {
		uint_t leaf, node, maxLevel, levels;
	}

	// {leaf_size, node_fanout, max_height, height}
	//TREE_PARMS[] = { {1,10,0xFF,3} };
	// Uncomment for NIST KATs only:
	TREE_PARMS[] = { {2,2,2,2}, {1,2,3,2}, {2,1,0xFF,3} };
#define TREE_PARM_CNT (sizeof(TREE_PARMS)/sizeof(TREE_PARMS[0]))

	//u08b_t msg[MAX_TREE_MSG_LEN+4], hashValCPU[MAX_TREE_MSG_LEN+4], hashValGPU[MAX_TREE_MSG_LEN+4];
	u08b_t *msg, *hashValCPU, *hashValGPU;
	uint_t i, j ,k, n, p, q;
	uint_t hashBits, node, leaf, leafBytes, msgBytes, byteCnt, levels, maxLevel;
	clock_t startTime, endTime;
	float cpuTime = 0.0;
	float gpuTime = 0.0;

	// Changed to use heap in case over 2 GB is ever desired
	msg =		 (u08b_t*) malloc( MAX_TREE_MSG_LEN + 4 );
	hashValCPU = (u08b_t*) malloc( MAX_TREE_MSG_LEN + 4 );
	hashValGPU = (u08b_t*) malloc( MAX_TREE_MSG_LEN + 4 );

	assert(blkSize == 256 || blkSize == 512 || blkSize == 1024);
	for (i = 0; i < MAX_TREE_MSG_LEN; i += 2) {   
		/* generate "incrementing" tree hash input msg data */
		msg[i  ] = (u08b_t) ((i ^ blkSize) ^ (i >> 16));
		msg[i+1] = (u08b_t) ((i ^ blkSize) >> 8);
	}

	for (k=q=n=0; k < HASH_BITS_CNT; k++) {
		hashBits = HASH_BITS[k];
		if (!Short_KAT_OK(blkSize, hashBits))
			continue;
		if ((verbose & V_KAT_SHORT) && (hashBits != blkSize))
			continue;
		for (p = 0; p < TREE_PARM_CNT; p++) {
			if (p && (verbose & V_KAT_SHORT))
				continue; /* keep short KATs short */
			if (p && hashBits != blkSize)
				continue; /* we only need one "non-full" size */

			leaf      = TREE_PARMS[p].leaf;
			node      = TREE_PARMS[p].node;
			maxLevel  = TREE_PARMS[p].maxLevel;
			levels    = TREE_PARMS[p].levels;
			leafBytes = (blkSize/8) << leaf;    /* number of bytes in a "full" leaf */

			/* different numbers of leaf results */
			for (j = 0; j < 4; j++) {
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

				printf("\n:Skein-512: %4d-bit hash, msgLen = %d bytes\n", hashBits, msgBytes);
				printf("\tTree: leaf=%02X, node=%02X, maxLevels=%02X\n", leaf, node, maxLevel );
				printf("\tRunning for %d iterations (excluding warmup pass)\n", ITERATIONS);

				//printf("\nMessage data:\n");
				//if (msgBytes == 0)
				//	printf("    (none)\n");
				//else
				//	ShowBytes(msgBytes, msg);

				// Untimed warmup pass for CPU
				SkeinTreeHash_CPU(blkSize, hashBits, msg, msgBytes, leaf, node, maxLevel, hashValCPU, skein_DebugFlag);

				startTime = clock();
				for( int it = 0; it < ITERATIONS; it++ ) {
					SkeinTreeHash_CPU(blkSize, hashBits, msg, msgBytes, leaf, node, maxLevel, hashValCPU, skein_DebugFlag);
				}
				endTime = clock();
				cpuTime = (float)(endTime - startTime) * 1000 / (float)CLOCKS_PER_SEC / ITERATIONS;

				printf("\nCPU Result:\n");
				ShowBytes((hashBits+7)/8,hashValCPU);
				printf( "\n\tHost computation took %.3f ms\n\n", cpuTime );
				printf("\t--------------------------------\n");

				// Untimed warmup pass for GPU
				SkeinTreeHash_GPU(blkSize, hashBits, msg, msgBytes, leaf, node, maxLevel, hashValGPU, skein_DebugFlag);

				startTime = clock();
				for( int it = 0; it < ITERATIONS; it++ ) {
					SkeinTreeHash_GPU(blkSize, hashBits, msg, msgBytes, leaf, node, maxLevel, hashValGPU, skein_DebugFlag);
				}
				endTime = clock();
				gpuTime = (float)(endTime - startTime) * 1000 / (float)CLOCKS_PER_SEC / ITERATIONS;

				printf("\nGPU Result:\n");
				ShowBytes((hashBits+7)/8,hashValGPU);
				uint_t diff = compareHashVals((hashBits+7)/8, hashValCPU, hashValGPU);
				if( diff ) {
					printf( "\nERROR: %d different bytes between host and device results!\n", diff);
				}
				printf( "\n\tDevice computation took %.3f ms\n\n", gpuTime );
				printf("*********************************************************\n");

			}

		}
	}

	// Cleanup
	free(msg);
	free(hashValCPU);
	free(hashValGPU);

} // Skein_GetKAT_Tree

/****************
 * Print usage
 ****************/
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

/*******************************
 *  Main function
 *******************************/
int main(int argc, char *argv[]) {

	int i;
	uint_t doKAT    =    0;   /* generate KAT vectors?    */
	uint_t doCustom =    0;   /* do custom long tree tests? */
	uint_t blkSize  =    0;   /* Skein state size in bits */
	uint_t maxLen   = 1024;   /* max block size   in bits */
	uint_t hashLen  =    0;   /* hash length      in bits (0 --> all) */
	uint_t seed0    = (uint_t) time(NULL); /* randomize based on time */
	uint_t oneBlk   =    0;   /* test block size */
	skein_DebugFlag = 0;

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
				case 'C': doCustom       = 1;                 break;
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

						default : skein_DebugFlag |= atoi(argv[i]+2); break;
					}
					break;
				
				default:  
					FatalError("Unsupported command-line option: %s", argv[i]);
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

	// NIST KATs if specified
	if (doKAT){
		Skein_GenKAT_Tree(blkSize);
	}

	// Custom long tree tests if specified
	if (doCustom) {
		Skein_Gen_Long_Trees(blkSize);
	}

	return 0;

} // main
