/***********************************************************************
**
** Implementation of the Skein block functions.
**
** Source code author: Doug Whiting, 2008.
**
** This algorithm and source code is released to the public domain.
**
************************************************************************/

#include <string.h>
#include "skein.h"

/* 64-bit rotate left */
u64b_t RotL_64(u64b_t x,uint_t N)
    {
    return (x << (N & 63)) | (x >> ((64-N) & 63));
    }

#define BLK_BITS    (WCNT*64)

/* macro to perform a key injection (same for all block sizes) */
#define InjectKey(r)                                                \
    for (i=0;i < WCNT;i++)                                          \
         X[i] += ks[((r)+i) % (WCNT+1)];                            \
    X[WCNT-3] += ts[((r)+0) % 3];                                   \
    X[WCNT-2] += ts[((r)+1) % 3];                                   \
    X[WCNT-1] += (r);                    /* avoid slide attacks */  \
    Skein_Show_Round(BLK_BITS,&ctx->h,SKEIN_RND_KEY_INJECT,X);


void Skein_512_Process_Block(Skein_512_Ctxt_t *ctx,const u08b_t *blkPtr,size_t blkCnt,size_t byteCntAdd)
    { /* do it in C */
    enum
        {
        WCNT = SKEIN_512_STATE_WORDS
        };

    size_t  i,r;
    u64b_t  ts[3];                            /* key schedule: tweak */
    u64b_t  ks[WCNT+1];                       /* key schedule: chaining vars */
    u64b_t  X [WCNT];                         /* local copy of vars */
    u64b_t  w [WCNT];                         /* local copy of input block */

    Skein_assert(blkCnt != 0);                /* never call with blkCnt == 0! */
    do  {
        /* this implementation only supports 2**64 input bytes (no carry out here) */
        ctx->h.T[0] += byteCntAdd;            /* update processed length */

        /* precompute the key schedule for this block */
        ks[WCNT] = SKEIN_KS_PARITY;
        for (i=0;i < WCNT; i++)
            {
            ks[i]     = ctx->X[i];
            ks[WCNT] ^= ctx->X[i];            /* compute overall parity */
            }
        ts[0] = ctx->h.T[0];
        ts[1] = ctx->h.T[1];
        ts[2] = ts[0] ^ ts[1];

        Skein_Get64_LSB_First(w,blkPtr,WCNT); /* get input block in little-endian format */
        Skein_Show_Block(BLK_BITS,&ctx->h,ctx->X,blkPtr,w,ks,ts);
        for (i=0;i < WCNT; i++)               /* do the first full key injection */
            {
            X[i]  = w[i] + ks[i];
            }
        X[WCNT-3] += ts[0];
        X[WCNT-2] += ts[1];

        Skein_Show_Round(BLK_BITS,&ctx->h,SKEIN_RND_KEY_INITIAL,X);
        for (r=1;r <= SKEIN_512_ROUNDS_TOTAL/8; r++)
            { /* unroll 8 rounds */
            X[0] += X[1]; X[1] = RotL_64(X[1],R_512_0_0); X[1] ^= X[0];
            X[2] += X[3]; X[3] = RotL_64(X[3],R_512_0_1); X[3] ^= X[2];
            X[4] += X[5]; X[5] = RotL_64(X[5],R_512_0_2); X[5] ^= X[4];
            X[6] += X[7]; X[7] = RotL_64(X[7],R_512_0_3); X[7] ^= X[6];  Skein_Show_Round(BLK_BITS,&ctx->h,8*r-7,X);

            X[2] += X[1]; X[1] = RotL_64(X[1],R_512_1_0); X[1] ^= X[2];
            X[4] += X[7]; X[7] = RotL_64(X[7],R_512_1_1); X[7] ^= X[4];
            X[6] += X[5]; X[5] = RotL_64(X[5],R_512_1_2); X[5] ^= X[6];
            X[0] += X[3]; X[3] = RotL_64(X[3],R_512_1_3); X[3] ^= X[0];  Skein_Show_Round(BLK_BITS,&ctx->h,8*r-6,X);

            X[4] += X[1]; X[1] = RotL_64(X[1],R_512_2_0); X[1] ^= X[4];
            X[6] += X[3]; X[3] = RotL_64(X[3],R_512_2_1); X[3] ^= X[6];
            X[0] += X[5]; X[5] = RotL_64(X[5],R_512_2_2); X[5] ^= X[0];
            X[2] += X[7]; X[7] = RotL_64(X[7],R_512_2_3); X[7] ^= X[2];  Skein_Show_Round(BLK_BITS,&ctx->h,8*r-5,X);

            X[6] += X[1]; X[1] = RotL_64(X[1],R_512_3_0); X[1] ^= X[6];
            X[0] += X[7]; X[7] = RotL_64(X[7],R_512_3_1); X[7] ^= X[0];
            X[2] += X[5]; X[5] = RotL_64(X[5],R_512_3_2); X[5] ^= X[2];
            X[4] += X[3]; X[3] = RotL_64(X[3],R_512_3_3); X[3] ^= X[4];  Skein_Show_Round(BLK_BITS,&ctx->h,8*r-4,X);
            InjectKey(2*r-1);

            X[0] += X[1]; X[1] = RotL_64(X[1],R_512_4_0); X[1] ^= X[0];
            X[2] += X[3]; X[3] = RotL_64(X[3],R_512_4_1); X[3] ^= X[2];
            X[4] += X[5]; X[5] = RotL_64(X[5],R_512_4_2); X[5] ^= X[4];
            X[6] += X[7]; X[7] = RotL_64(X[7],R_512_4_3); X[7] ^= X[6];  Skein_Show_Round(BLK_BITS,&ctx->h,8*r-3,X);

            X[2] += X[1]; X[1] = RotL_64(X[1],R_512_5_0); X[1] ^= X[2];
            X[4] += X[7]; X[7] = RotL_64(X[7],R_512_5_1); X[7] ^= X[4];
            X[6] += X[5]; X[5] = RotL_64(X[5],R_512_5_2); X[5] ^= X[6];
            X[0] += X[3]; X[3] = RotL_64(X[3],R_512_5_3); X[3] ^= X[0];  Skein_Show_Round(BLK_BITS,&ctx->h,8*r-2,X);

            X[4] += X[1]; X[1] = RotL_64(X[1],R_512_6_0); X[1] ^= X[4];
            X[6] += X[3]; X[3] = RotL_64(X[3],R_512_6_1); X[3] ^= X[6];
            X[0] += X[5]; X[5] = RotL_64(X[5],R_512_6_2); X[5] ^= X[0];
            X[2] += X[7]; X[7] = RotL_64(X[7],R_512_6_3); X[7] ^= X[2];  Skein_Show_Round(BLK_BITS,&ctx->h,8*r-1,X);

            X[6] += X[1]; X[1] = RotL_64(X[1],R_512_7_0); X[1] ^= X[6];
            X[0] += X[7]; X[7] = RotL_64(X[7],R_512_7_1); X[7] ^= X[0];
            X[2] += X[5]; X[5] = RotL_64(X[5],R_512_7_2); X[5] ^= X[2];
            X[4] += X[3]; X[3] = RotL_64(X[3],R_512_7_3); X[3] ^= X[4];  Skein_Show_Round(BLK_BITS,&ctx->h,8*r  ,X);
            InjectKey(2*r);
            }
        /* do the final "feedforward" xor, update context chaining vars */
        for (i=0;i < WCNT;i++)
            ctx->X[i] = X[i] ^ w[i];
        Skein_Show_Round(BLK_BITS,&ctx->h,SKEIN_RND_FEED_FWD,ctx->X);

		Skein_Clear_First_Flag(ctx->h);		/* clear the start bit */
        blkPtr += SKEIN_512_BLOCK_BYTES;
        }
    while (--blkCnt);
    }

#if defined(SKEIN_CODE_SIZE) || defined(SKEIN_PERF)
size_t Skein_512_Process_Block_CodeSize(void)
    {
    return ((u08b_t *) Skein_512_Process_Block_CodeSize) -
           ((u08b_t *) Skein_512_Process_Block);
    }
uint_t Skein_512_Unroll_Cnt(void)
    {
    return 1;
    }
#endif
