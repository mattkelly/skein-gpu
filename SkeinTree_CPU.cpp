#include "SkeinTest.h"

/* 
 * Generate a KAT test for the given data and tree parameters.
 * This is an "all-in-one" call. It is not intended to represent
 * how a real multi-processor version would be implemented, but
 * the results will be the same
 */
void SkeinTreeHash_CPU( uint_t blkSize, uint_t hashBits, const u08b_t *msg, size_t msgBytes,
	uint_t leaf, uint_t node, uint_t maxLevel, u08b_t *hashRes) {

	enum      { MAX_HEIGHT = 32 }; /* how deep we can go here */
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

		// lower levels of tree (not final level)
		nodeLen = blkBytes << ((height) ? node : leaf);
		for (srcOffs=dstOffs=0; srcOffs <= bCnt; ) {
			n = bCnt - srcOffs;         /* number of bytes left at this level */
			if (n > nodeLen)            /* limit to node size */
				n = nodeLen;
			s = G;
			s.u.h.T[0] = srcOffs;       /* nonzero initial offset in tweak! */
			Skein_Set_Tree_Level(s.u.h,height+1);
			Skein_Update(&s, M+srcOffs, n*8);
			Skein_512_Final_Pad(&s.u.ctx_512, M+dstOffs);  /* finish up this node, output intermediate result to M[]*/
			dstOffs+=blkBytes;
			srcOffs+=n;
			if (srcOffs >= bCnt) /* special logic to handle (msgBytes == 0) case */
				break;
		}
		bCnt = dstOffs;

	} // walk up tree

	/* output the result */
	Skein_512_Output(&s.u.ctx_512, hashRes);

} // Skein_TreeHash
