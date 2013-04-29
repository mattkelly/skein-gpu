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

#ifndef SKEIN_DEBUG
uint_t skein_DebugFlag; /* dummy flags (if not defined elsewhere) */
#endif

#define SKEIN_DEBUG_SHORT   (SKEIN_DEBUG_HDR | SKEIN_DEBUG_STATE | SKEIN_DEBUG_TWEAK | SKEIN_DEBUG_KEY | SKEIN_DEBUG_INPUT_08 | SKEIN_DEBUG_FINAL)
#define SKEIN_DEBUG_DEFAULT (SKEIN_DEBUG_SHORT)

#define MAX_TREE_MSG_LEN  (1 << 12)

/****************************
 * Debug IO helper routines
 ****************************/
// Print out a msg and exit with an error code
void FatalError(const char *s, ...);

// Formatted output of byte array
void ShowBytes(uint_t cnt, const u08b_t *b);

void Show_Debug(const char *s, ...);

/***********************************************************************/
/* An AHS-like API that allows explicit setting of block size          */
/*    [i.e., the AHS API selects a block size based solely on the ]    */
/*    [hash result length, while Skein allows independent hash    ]    */
/*    [result size and block size                                 ]    */
/***********************************************************************/

/* process data to be hashed */
int Skein_Update(hashState *state, const BitSequence *data, DataLength databitlen);

/* filter out <blkSize,hashBits> pairs in short KAT mode */
uint_t Short_KAT_OK(uint_t blkSize,uint_t hashBits);

/* 
 * Generate a KAT test for the given data and tree parameters.
 * Runs on CPU
 */
void SkeinTreeHash_CPU( uint_t blkSize, uint_t hashBits, const u08b_t *msg, size_t msgBytes,
	uint_t leaf, uint_t node, uint_t maxLevel, u08b_t *hashRes);

/* 
 * Generate a KAT test for the given data and tree parameters.
 * Runs on GPU
 */
int SkeinTreeHash_GPU( uint_t blkSize, uint_t hashBits, const u08b_t *msg, size_t msgBytes,
	uint_t leaf, uint_t node, uint_t maxLevel, u08b_t *hashRes );

/*
 * Generate tree-mode hash KAT vectors.
 */
void Skein_GenKAT_Tree(uint_t blkSize);
