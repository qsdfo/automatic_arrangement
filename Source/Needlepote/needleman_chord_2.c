/**
*
*  @file			needleman.c
*  @brief			Optimized Needleman-Wunsch alignment with homopolymers handling
*
*	This file contains the implementation of the optimized Needleman-Wunsch alignment algorithm with homopolymers handling
*
*  @author			Philippe Esling
*	@version		1.1
*	@date			16-01-2013
*  @copyright		UNIGE - GEN/EV (Pawlowski) - 2013
*	@licence		MIT Media Licence
*
*/

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#define U_FEPS 1.192e-6F          /* 1.0F + E_FEPS != 1.0F */
#define U_DEPS 2.22e-15           /* 1.0 +  E_DEPS != 1.0  */

#define E_FPEQ(a,b,e) (((b - e) < a) && (a < (b + e)))

#define _DIAG 	0
#define _UP  	1
#define _LEFT 	2

#define BUFFSIZE	4096
#define SEQSIZE		1024

#define PITCH_DIM     5
#define NUM_PITCH_CLASS     12

// Debug
#define LEN 10
#define BUF_SIZE 39

int		DNAFull[26][26] =
{{5,-4,-4,-1,0,0,-4,-1,0,0,-4,0,1,-2,0,0,0,1,-4,-4,-4,-1,1,0,-4,0},
{-4,-1,-1,-2,0,0,-1,-2,0,0,-1,0,-3,-1,0,0,0,-3,-1,-1,-1,-2,-3,0,-1,0},
{-4,-1,5,-4,0,0,-4,-1,0,0,-4,0,1,-2,0,0,0,-4,1,-4,-4,-1,-4,0,1,0},
{-1,-2,-4,-1,0,0,-1,-2,0,0,-1,0,-3,-1,0,0,0,-1,-3,-1,-1,-2,-1,0,-3,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{-4,-1,-4,-1,0,0,5,-4,0,0,1,0,-4,-2,0,0,0,1,1,-4,-4,-1,-4,0,-4,0},
{-1,-2,-1,-2,0,0,-4,-1,0,0,-3,0,-1,-1,0,0,0,-3,-3,-1,-1,-2,-1,0,-1,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{-4,-1,-4,-1,0,0,1,-3,0,0,-1,0,-4,-1,0,0,0,-2,-2,1,1,-3,-2,0,-2,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{1,-3,1,-3,0,0,-4,-1,0,0,-4,0,-1,-1,0,0,0,-2,-2,-4,-4,-1,-2,0,-2,0},
{-2,-1,-2,-1,0,0,-2,-1,0,0,-1,0,-1,-1,0,0,0,-1,-1,-2,-2,-1,-1,0,-1,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{1,-3,-4,-1,0,0,1,-3,0,0,-2,0,-2,-1,0,0,0,-1,-2,-4,-4,-1,-2,0,-4,0},
{-4,-1,1,-3,0,0,1,-3,0,0,-2,0,-2,-1,0,0,0,-2,-1,-4,-4,-1,-4,0,-2,0},
{-4,-1,-4,-1,0,0,-4,-1,0,0,1,0,-4,-2,0,0,0,-4,-4,5,5,-4,1,0,1,0},
{-4,-1,-4,-1,0,0,-4,-1,0,0,1,0,-4,-2,0,0,0,-4,-4,5,5,-4,1,0,1,0},
{-1,-2,-1,-2,0,0,-1,-2,0,0,-3,0,-1,-1,0,0,0,-1,-1,-4,-4,-1,-3,0,-3,0},
{1,-3,-4,-1,0,0,-4,-1,0,0,-2,0,-2,-1,0,0,0,-2,-4,1,1,-3,-1,0,-2,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
{-4,-1,1,-3,0,0,-4,-1,0,0,-2,0,-2,-1,0,0,0,-4,-2,1,1,-3,-2,0,-1,0},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

char *int2bin(int a, char *buffer, int buf_size) {
    buffer += (buf_size - 1);

    for (int i = 31; i >= 0; i--) {
        *buffer-- = (a & 1) + '0';

        a >>= 1;
    }

    return buffer;
}

float score_chord(int a, int b){
    int score = 0;

    char buffer[BUF_SIZE];
    buffer[BUF_SIZE - 1] = '\0';

    // score = number of bit on in the mask
    for (int i=0; i < NUM_PITCH_CLASS; i++)
    {
        int2bin(a, buffer, BUF_SIZE - 1);
        printf("Binary representation of a : %s", buffer);
        score += (a & 0x1) & (b & 0x1);
        a = a >> 2;
        b = b >> 2;
    }

    return score;
}

static float getScore(const int *horizGap_m, const int *vertGap_m, const int *m, int lena, int lenb, int *start1, int *start2, int noEndGap_n)
{
    int i,j, cursor;
    float score = INT_MIN;
    *start1 = lena-1;
    *start2 = lenb-1;

    if(noEndGap_n)
    {
        cursor = lena * lenb - 1;
        if(m[cursor]>horizGap_m[cursor]&&m[cursor]>vertGap_m[cursor])
        score = m[cursor];
        else if(horizGap_m[cursor]>vertGap_m[cursor])
        score = horizGap_m[cursor];
        else
        score = vertGap_m[cursor];
    }
    else {

        for (i = 0; i < lenb; ++i)
        {
            cursor = (lena - 1) * lenb + i;
            if(m[cursor]>score)
            {
                *start2 = i;
                score = m[cursor];
            }
            if(horizGap_m[cursor]>score)
            {
                score = horizGap_m[cursor];
                *start2 = i;
            }
            if(vertGap_m[cursor]>score)
            {
                score = vertGap_m[cursor];
                *start2 = i;
            }
        }

        for (j = 0; j < lena; ++j)
        {
            cursor = j * lenb + lenb - 1;
            if(m[cursor]>score)
            {
                *start1 = j;
                *start2 = lenb-1;
                score = m[cursor];
            }
            if(horizGap_m[cursor]>score)
            {
                score = horizGap_m[cursor];
                *start1 = j;
                *start2 = lenb-1;
            }
            if(vertGap_m[cursor]>score)
            {
                score = vertGap_m[cursor];
                *start1 = j;
                *start2 = lenb-1;
            }
        }
    }
    return score;
}

void homopolHandling(int type, int a, int b, int homoPol, int *homopolState, char *matching, int *stats, int addDist, int nbAligns)
{
    int		i,j;

    switch (type)
    {
        case _UP:
        if (a == homopolState[0])
        {
            homopolState[1] += 1;
            homopolState[2] += addDist;
            homopolState[5] += addDist;
        }
        else
        {
            if (homopolState[1] >= homoPol && homopolState[4] >= homoPol && homopolState[2] != 0)
            {
                stats[4] -= homopolState[2];
                stats[0] += homopolState[2];
                for (i = 2; i < 2 + homopolState[1] && nbAligns - i >= 0; i++)
                matching[nbAligns - i] = 'h';
                homopolState[4] -= 10e5;
            }
            homopolState[0] = a;
            homopolState[1] = 1;
            homopolState[2] = addDist;
            if (homopolState[4] >= homoPol && homopolState[1] >= homoPol && homopolState[5] != 0)
            {
                stats[4] -= homopolState[5];
                stats[0] += homopolState[5];
                for (i = 2; i < i + homopolState[4] && nbAligns - i >= 0; i++)
                matching[nbAligns - i] = 'h';
                homopolState[1] -= 10e5;
            }
            homopolState[3] = '-';
            homopolState[4] = 0;
            homopolState[5] = addDist;

        }
        break;
        case _LEFT:
        if (b == homopolState[3])
        {
            homopolState[2] += addDist;
            homopolState[4] += 1;
            homopolState[5] += addDist;
        }
        else
        {
            if (homopolState[4] >= homoPol && homopolState[1] >= homoPol && homopolState[5] != 0)
            {
                stats[4] -= homopolState[5];
                stats[0] += homopolState[5];
                for (j = 2; j < 2 + homopolState[4] && nbAligns - j >= 0; j++)
                matching[nbAligns - j] = 'h';
                homopolState[1] -= 10e5;
            }
            homopolState[3] = b;
            homopolState[4] = 1;
            homopolState[5] = addDist;
            if (homopolState[1] >= homoPol && homopolState[4] >= homoPol && homopolState[2] != 0)
            {
                stats[4] -= homopolState[2];
                stats[0] += homopolState[2];
                for (j = 2; j < 2 + homopolState[1] && nbAligns - j >= 0; j++)
                matching[nbAligns - j] = 'h';
                homopolState[4] -= 10e5;
            }
            homopolState[0] = '-';
            homopolState[1] = 0;
            homopolState[2] = addDist;
        }
        break;
        case _DIAG:
        if (homoPol > 0)
        {
            if (homopolState[0] != a)
            {
                if (homopolState[1] >= homoPol && homopolState[4] >= homoPol && homopolState[2] != 0)
                {
                    stats[4] -= homopolState[2];
                    stats[0] += homopolState[2];
                    for (j = 2; j < 2 + homopolState[1] && nbAligns - j >= 0; j++)
                    matching[nbAligns - j] = 'h';
                    homopolState[4] -= 10e5;
                }
                homopolState[0] = a;
                homopolState[1] = 1;
                homopolState[2] = 0;
            }
            else
            {
                if (a == b)
                homopolState[1] += 1;
                else
                {
                    if (homopolState[1] >= homoPol && homopolState[4] >= homoPol && homopolState[2] != 0)
                    {
                        stats[4] -= homopolState[2];
                        stats[0] += homopolState[2];
                        for (j = 2; j < 2 + homopolState[1] && nbAligns - j >= 0; j++)
                        matching[nbAligns - j] = 'h';
                        homopolState[4] -= 10e5;
                    }
                    homopolState[1] -= 10e5;
                    homopolState[4] -= 10e5;
                }
            }
            if (homopolState[3] != b)
            {
                if (homopolState[4] >= homoPol && homopolState[1] >= homoPol && homopolState[5] > 0)
                {
                    stats[4] -= homopolState[5];
                    stats[0] += homopolState[5];
                    for (j = 2; j < 2 + homopolState[4] && nbAligns - j >= 0; j++)
                    matching[nbAligns - j] = 'h';
                    homopolState[1] -= 10e5;
                }
                homopolState[3] = b;
                homopolState[4] = 1;
                homopolState[5] = 0;
            }
            else
            {
                if (a == b)
                homopolState[4] += 1;
                else
                {
                    if (homopolState[4] >= homoPol && homopolState[1] >= homoPol && homopolState[5] > 0)
                    {
                        stats[4] -= homopolState[5];
                        stats[0] += homopolState[5];
                        for (j = 2; j < 2 + homopolState[4] && nbAligns - j >= 0; j++)
                        matching[nbAligns - j] = 'h';
                        homopolState[1] -= 10e5;
                    }
                    homopolState[4] -= 10e5;
                    homopolState[1] -= 10e5;
                }
            }
        }
        break;
        default:
        return;
    }
}

float needlemanWunsch(const int *a, const int *b, int noEndGap_n, int distthresh, int homoPol, int gapopen, int gapextend,
    int endgapopen, int endgapextend, int lena, int lenb, int *m, int *horizGap_m, int *vertGap_m, int *trBack,
    int verbose, const char *fOut, int dif, const char *seqID1, const char *seqID2, int oneGap, float cut){
        int		xpos, ypos;
        int		bconvcode;
        int		match;
        int		horizGap_mp;
        int		vertGap_mp;
        int		mp;
        int		i, j;
        int		cursor = 0, cursorp;
        int		*start1, *start2;
        int		nbAligns = 0;
        char	*recon1 = NULL, *recon2 = NULL;
        char	*matching = NULL;
        FILE	*outputFile;
        int		testog;
        int		testeg;
        int		lastGap = 0;
        int		nbEndGap = 0;
        int     bestSoFar;
        /* Align stats : nbId, F, nbGaps, F, nbDiffs, F */
        int		stats[6] = {0, 0, 0, 0, 0, 0};
        int 	homopolState[6] = {'?', 0, 0, '?', 0, 0};

        if (noEndGap_n == 1)
        {
            endgapopen = 0;
            endgapextend = 0;
        }
        start1 = calloc(1, sizeof(int));
        start2 = calloc(1, sizeof(int));
        horizGap_m[0] = -endgapopen-gapopen;
        vertGap_m[0] = -endgapopen-gapopen;


        // m[0] = DNAFull[a[0] - 'A'][b[0] - 'A'];
        m[0] = score_chord(a[0],b[0]);

        /* First initialise the first column */
        for (ypos = 1; ypos < lena; ++ypos)
        {
            // match = DNAFull[a[ypos] - 'A'][b[0] - 'A'];
            match = score_chord(a[ypos],b[0]);
            cursor = ypos * lenb;
            cursorp = cursor - lenb;
            testog = m[cursorp] - gapopen;
            testeg = vertGap_m[cursorp] - gapextend;
            vertGap_m[cursor] = (testog >= testeg ? testog : testeg);
            m[cursor] = match - (endgapopen + (ypos - 1) * endgapextend);
            horizGap_m[cursor] = - endgapopen - ypos * endgapextend - gapopen;
        }
        horizGap_m[cursor] -= endgapopen - gapopen;
        for (xpos = 1; xpos < lenb; ++xpos)
        {
            // match = DNAFull[a[0] - 'A'][b[xpos] - 'A'];
            match = score_chord(a[0],b[xpos]);
            cursor = xpos;
            cursorp = xpos -1;
            testog = m[cursorp] - gapopen;
            testeg = horizGap_m[cursorp] - gapextend;
            horizGap_m[cursor] = (testog >= testeg ? testog : testeg);
            m[cursor] = match - (endgapopen + (xpos - 1) * endgapextend);
            vertGap_m[cursor] = -endgapopen - xpos * endgapextend -gapopen;
        }
        vertGap_m[cursor] -= endgapopen - gapopen;
        xpos = 1;
        /*
        * Filling step
        */
        while (xpos != lenb)
        {
            ypos = 1;
            bconvcode = b[xpos];
            cursorp = xpos-1;
            cursor = xpos++;
            bestSoFar = INT_MAX;
            while (ypos < lena)
            {
                match = score_chord(a[ypos++],bconvcode);
                cursor += lenb;
                mp = m[cursorp];
                horizGap_mp = horizGap_m[cursorp];
                vertGap_mp = vertGap_m[cursorp];
                if(mp > horizGap_mp && mp > vertGap_mp)
                m[cursor] = mp+match;
                else if(horizGap_mp > vertGap_mp)
                m[cursor] = horizGap_mp+match;
                else
                m[cursor] = vertGap_mp+match;
                if(xpos==lenb)
                {
                    testog = m[++cursorp] - endgapopen;
                    testeg = vertGap_m[cursorp] - endgapextend;
                }
                else
                {
                    testog = m[++cursorp];
                    if (testog<horizGap_m[cursorp])
                    testog = horizGap_m[cursorp];
                    testog -= gapopen;
                    testeg = vertGap_m[cursorp] - gapextend;
                }
                if(testog > testeg)
                vertGap_m[cursor] = testog;
                else
                vertGap_m[cursor] = testeg;
                cursorp += lenb;
                if(ypos==lena)
                {
                    testog = m[--cursorp] - endgapopen;
                    testeg = horizGap_m[cursorp] - endgapextend;
                }
                else
                {
                    testog = m[--cursorp];
                    if (testog<vertGap_m[cursorp])
                    testog = vertGap_m[cursorp];
                    testog -= gapopen;
                    testeg = horizGap_m[cursorp] - gapextend;
                }
                if (testog > testeg)
                horizGap_m[cursor] = testog;
                else
                horizGap_m[cursor] = testeg;
            }
        }
        getScore(horizGap_m, vertGap_m, m, lena, lenb, start1, start2, noEndGap_n);
        xpos = *start2;
        ypos = *start1;
        cursorp = 0;
        /*
        * Trace-back step
        */
        while (xpos>=0 && ypos>=0)
        {
            cursor = ypos*lenb+xpos;
            mp = m[cursor];
            if(cursorp == _LEFT && E_FPEQ((ypos==0||(ypos==lena)?
            endgapextend:gapextend), (horizGap_m[cursor]-horizGap_m[cursor+1]),U_FEPS))
            {
                trBack[cursor] = _LEFT;
                xpos--;
            }
            else if(cursorp== _UP && E_FPEQ((xpos==0||(xpos==lenb)?
            endgapextend:gapextend), (vertGap_m[cursor]-vertGap_m[cursor+lenb]),U_FEPS))
            {
                trBack[cursor] = _UP;
                ypos--;
            }
            else if(mp >= horizGap_m[cursor] && mp >= vertGap_m[cursor])
            {
                if(cursorp == _LEFT && E_FPEQ(mp,horizGap_m[cursor],U_FEPS))
                {
                    trBack[cursor] = _LEFT;
                    xpos--;
                }
                else if(cursorp == _UP && E_FPEQ(mp,vertGap_m[cursor],U_FEPS))
                {
                    trBack[cursor] = _UP;
                    ypos--;
                }
                else
                {
                    trBack[cursor] = 0;
                    ypos--;
                    xpos--;
                }
            }
            else if(horizGap_m[cursor]>=vertGap_m[cursor] && xpos>-1)
            {
                trBack[cursor] = _LEFT;
                xpos--;
            }
            else if(ypos>-1)
            {
                trBack[cursor] = _UP;
                ypos--;
            }
            cursorp = trBack[cursor];
        }
        xpos = *start2;
        ypos = *start1;
        if (verbose != 0)
        {
            recon1 = malloc((lenb + lena) * sizeof(char));
            recon2 = malloc((lenb + lena) * sizeof(char));
            matching = malloc((lenb + lena) * sizeof(char));
        }
        for (i = lenb - 1; i > xpos;)
        {
            stats[2]++;
            stats[4] += 1 - noEndGap_n;
            nbEndGap++;
            /*		stats[0] += noEndGap_n;*/
            if (verbose)
            {
                recon1[nbAligns] = '-';
                recon2[nbAligns] = b[i];
                matching[nbAligns] = (noEndGap_n ? 'e' : ' ');
            }
            nbAligns++; i--;
            if (homoPol > 0)
            homopolHandling(_LEFT,b[i + 1],b[i + 1], homoPol, homopolState, matching, stats, noEndGap_n, nbAligns);
        }
        for (j = lena - 1; j > ypos;)
        {
            stats[2]++;
            stats[4] += 1 - noEndGap_n;
            nbEndGap++;
            if (verbose)
            {
                recon1[nbAligns] = a[j];
                recon2[nbAligns] = '-';
                matching[nbAligns] = (noEndGap_n ? 'e' : ' ');
            }
            j--; nbAligns++;
            if (homoPol > 0)
            homopolHandling(_UP,a[j + 1],a[j + 1], homoPol, homopolState, matching, stats, noEndGap_n, nbAligns);
        }
        while (xpos >= 0 && ypos >= 0)
        {
            cursor = ypos * lenb + xpos;
            switch (trBack[cursor])
            {
                case _DIAG:
                lastGap = 0;
                mp = (a[ypos] == b[xpos]);
                stats[0] += (mp ? 1 : 0);
                stats[4] += (mp ? 0 : 1);
                if (verbose)
                {
                    recon1[nbAligns] = a[ypos];
                    recon2[nbAligns] = b[xpos];
                    matching[nbAligns] = (mp ? '|' : '.');
                }
                ypos--; xpos--; nbAligns++;
                if (homoPol > 0)
                homopolHandling(_DIAG,a[ypos + 1],b[xpos + 1], homoPol, homopolState, matching, stats, mp, nbAligns);
                break;
                case _LEFT:
                stats[2]++;
                /*stats[0] += (ypos == (lena - 1)	&& noEndGap_n ? 1 : 0);*/
                stats[4] += (ypos == (lena - 1)	&& noEndGap_n ? 0 : (lastGap && oneGap ? 0 : 1));
                if (ypos == (lena - 1))
                nbEndGap++;
                if (verbose)
                {
                    recon1[nbAligns] = '-';
                    recon2[nbAligns] = b[xpos];
                    matching[nbAligns] = (ypos == (lena - 1) && noEndGap_n ? 'e' : (lastGap && oneGap ? 'c' : ' '));
                }
                xpos--; nbAligns++;
                lastGap = 1;
                if (homoPol > 0)
                homopolHandling(_LEFT,b[xpos + 1],b[xpos + 1], homoPol, homopolState, matching, stats, (lastGap && oneGap ? 0 : 1), nbAligns);
                break;
                case _UP:
                stats[2]++;
                /*stats[0] += (xpos == (lenb - 1) && noEndGap_n ? 1 : 0);*/
                stats[4] += (xpos == (lenb - 1) && noEndGap_n ? 0 : (lastGap && oneGap ? 0 : 1));
                if (xpos == (lenb - 1))
                nbEndGap++;
                if (verbose)
                {
                    recon1[nbAligns] = a[ypos];
                    recon2[nbAligns] = '-';
                    matching[nbAligns] = (xpos == (lenb - 1) && noEndGap_n ? 'e' : (lastGap && oneGap ? 'c' : ' '));
                }
                ypos--; nbAligns++;
                lastGap = 1;
                if (homoPol > 0)
                homopolHandling(_UP,a[ypos + 1],a[ypos + 1], homoPol, homopolState, matching, stats, (lastGap && oneGap ? 0 : 1), nbAligns);
                break;
                default:
                break;
            }
        }
        for (; xpos >= 0 ; xpos--)
        {
            stats[2]++;
            stats[4] += 1 - noEndGap_n;
            nbEndGap++;
            /*stats[0] += noEndGap_n;*/
            if (verbose)
            {
                recon1[nbAligns] = '-';
                recon2[nbAligns] = b[xpos];
                matching[nbAligns] = (noEndGap_n ? 'e' : ' ');
            }
            nbAligns++;
            if (homoPol > 0)
            homopolHandling(_LEFT,b[xpos],b[xpos], homoPol, homopolState, matching, stats, 1 - noEndGap_n, nbAligns);
        }
        for (; ypos >= 0; ypos--)
        {
            stats[2]++;
            stats[4] += 1 - noEndGap_n;
            nbEndGap++;
            /*stats[0] += noEndGap_n;*/
            if (verbose)
            {
                recon1[nbAligns] = a[ypos];
                recon2[nbAligns] = '-';
                matching[nbAligns] = (noEndGap_n ? 'e' : ' ');
            }
            nbAligns++;
            if (homoPol > 0)
            homopolHandling(_UP,a[ypos],a[ypos], homoPol, homopolState, matching, stats, 1 - noEndGap_n, nbAligns);
        }
        int tstOut = (100.0f * nbEndGap / nbAligns) <= cut && (100.0f - ((100.0f * stats[4]) / nbAligns)) >= distthresh;
        if (verbose == 1 && tstOut)
        {
            int i,j;
            outputFile = fopen(fOut, "a+");
            fprintf(outputFile, "**************************\n");
            fprintf(outputFile, "Seq1\t: %s\n", seqID1);
            fprintf(outputFile, "Seq2\t: %s\n", seqID2);
            fprintf(outputFile, "Length \t\t: %d\n", nbAligns);
            fprintf(outputFile, "Identities \t: %d (%.2f%%)\n", stats[0], 100.0f * (float)stats[0]/nbAligns);
            fprintf(outputFile, "Diffs (filt) \t: %d (%.2f%%)\n", stats[4], 100.0f * (float)stats[4]/nbAligns);
            fprintf(outputFile, "Gaps \t\t: %d (%.2f%%)\n", stats[2], 100.0f * (float)stats[2]/nbAligns);
            fprintf(outputFile, "\n");
            for (j = nbAligns - 1; j >= 0; j -= 50)
            {
                for (i = j; i > j - 50 && i >= 0; i--)
                fprintf(outputFile, "%c", recon1[i]);
                fprintf(outputFile, "\n");
                for (i = j; i > j - 50 && i >= 0; i--)
                fprintf(outputFile, "%c", matching[i]);
                fprintf(outputFile, "\n");
                for (i = j; i > j - 50 && i >= 0; i--)
                fprintf(outputFile, "%c", recon2[i]);
                fprintf(outputFile, "\n\n");
            }
            fclose(outputFile);
        }
        if (verbose)
        {
            free(matching);
            free(recon1);
            free(recon2);
        }
        free(start1);
        free(start2);
        if (!tstOut)
        {
            return -1;
        }
        if (dif)
        {
            return stats[4];
        }
        return 100.0f * stats[4] / nbAligns;
    }

    static PyObject	*needleman_chord(PyObject* self, PyObject* args)
    {
        int *a;
        int *b;
        int lena;
        int lenb;
        int noEndGap_n = 1;
        int distthresh = 1e9;
        int homoPol = 0;
        int gapopen = -3;
        int gapextend = -1;
        int endgapopen = 0;
        int endgapextend = 0;
        int verbose = 0;
        const char *fOut = NULL;
        int dif = 0;
        const char *seqID1 = NULL;
        const char *seqID2 = NULL;
        int oneGap = 0;
        float cut = 1.0;

        if (!PyArg_ParseTuple(args, "iiii", &a, &b, &lena, &lenb))
        {
            printf("Bad format for argument")
            return NULL;
        }

        int curMalloc = lena * lenb;
        int *m = malloc(curMalloc * sizeof(int));
        int *horizGap_m = malloc(curMalloc * sizeof(int));
        int *vertGap_m = malloc(curMalloc * sizeof(int));
        int *trBack = malloc(curMalloc * sizeof(int));

        return Py_BuildValue("i", needlemanWunsch(a, b, noEndGap_n, distthresh, homoPol, gapopen, gapextend,
            endgapopen, endgapextend, lena, lenb, m, horizGap_m, vertGap_m, trBack,
            verbose, fOut, dif, seqID1, seqID2, oneGap, cut));
        }

        static PyMethodDef NeedleMethods[] = {
            {"needleman_chord", (PyCFunction)needleman_chord, METH_VARARGS, "Calculate Needleman-Wunsch for a whole set"},
            {NULL, NULL, 0, NULL}
        };

        PyMODINIT_FUNC  // Precise return type = void + declares any special linkage declarations required by the platform
        initneedleman_chord(void)
        {
            PyObject *module
            module = Py_InitModule("needleman_chord", NeedleMethods);

            if (module == NULL)
                return;
        }
