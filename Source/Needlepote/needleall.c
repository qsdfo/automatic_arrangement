#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include "needleman.h"

#define U_FEPS 1.192e-6F          /* 1.0F + E_FEPS != 1.0F */
#define U_DEPS 2.22e-15           /* 1.0 +  E_DEPS != 1.0  */

#define E_FPEQ(a,b,e) (((b - e) < a) && (a < (b + e)))

#define _DIAG 	0
#define _UP  	1
#define _LEFT 	2

#define BUFFSIZE	4096
#define SEQSIZE		1024

FILE		*fastaOpen(const char *filename);
int 		fastaIterate(FILE *handle, char *seq, char *id, char *buff);
int 		fastaSkip(FILE *handle, int nSkip, char *buff);

static float getScore(const int *horizGap_m, const int *vertGap_m, const int *m, int lena, int lenb,
					  int *start1, int *start2, int noEndGap)
{
    int i,j, cursor;
    float score = INT_MIN;
    *start1 = lena-1;
    *start2 = lenb-1;

    if(noEndGap)
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

void homopolHandling(int type, char a, char b, int homoPol, int *homopolState, char *matching, int *stats, int addDist, int nbAligns)
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

float _needleAffine(const char *a, const char *b, int noEndGap, int distthresh, int homoPol, int gapopen, int gapextend,
		int endgapopen, int endgapextend, int lena, int lenb, int *m, int *horizGap_m, int *vertGap_m, int *trBack,
		int verbose, char *fOut, int dif, char *seqID1, char *seqID2, int oneGap, float cut)
{
    int		xpos, ypos;
    int		bconvcode;
    int		match;
    int		horizGap_mp;
    int		vertGap_mp;
    int		mp;
	int		i, j;
    int		cursor, cursorp;
    int		*start1, *start2;
	int		nbAligns = 0;
    char	*recon1, *recon2;
	char	*matching;
    FILE	*outputFile;
    int		testog;
    int		testeg;
    int		score = 0;
	int		lastGap = 0;
	int		nbEndGap = 0;
	/* Align stats : nbId, F, nbGaps, F, nbDiffs, F */
	int		stats[6] = {0, 0, 0, 0, 0, 0};
    int 	homopolState[6] = {'?', 0, 0, '?', 0, 0};

    if (noEndGap == 1)
    {
    	endgapopen = 0;
    	endgapextend = 0;
    }
	start1 = calloc(1, sizeof(int));
	start2 = calloc(1, sizeof(int));
    horizGap_m[0] = -endgapopen-gapopen;
    vertGap_m[0] = -endgapopen-gapopen;
    m[0] = DNAFull[a[0] - 'A'][b[0] - 'A'];
    /* First initialise the first column */
    for (ypos = 1; ypos < lena; ++ypos)
    {
    	match = DNAFull[a[ypos] - 'A'][b[0] - 'A'];
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
    	match = DNAFull[a[0] - 'A'][b[xpos] - 'A'];
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
        bconvcode = b[xpos] - 'A';
        cursorp = xpos-1;
        cursor = xpos++;
        while (ypos < lena)
        {
			match = DNAFull[a[ypos++] - 'A'][bconvcode];
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
	getScore(horizGap_m, vertGap_m, m, lena, lenb, start1, start2, noEndGap);
	xpos = *start2;
	ypos = *start1;
    cursorp=0;
    cursor= 1;
	/*
	 * Trace-back step
	 */
	while (xpos>=0 && ypos>=0)
    {
    	cursor = ypos*lenb+xpos;
    	mp = m[cursor];
    	if(cursorp == _LEFT && E_FPEQ((ypos==0||(ypos==lena-1)?
    			endgapextend:gapextend), (horizGap_m[cursor]-horizGap_m[cursor+1]),U_FEPS))
    	{
    		trBack[cursor] = _LEFT;
    		xpos--;
    	}
    	else if(cursorp== _UP && E_FPEQ((xpos==0||(xpos==lenb-1)?
    			endgapextend:gapextend), (vertGap_m[cursor]-vertGap_m[cursor+lenb]),U_FEPS))
    	{
    		trBack[cursor] = _UP;
    		ypos--;
    	}
    	else if(mp >= horizGap_m[cursor] && mp>= vertGap_m[cursor])
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
    /*if (verbose != 0)
    {*/
    	recon1 = malloc((lenb + lena) * sizeof(char));
    	recon2 = malloc((lenb + lena) * sizeof(char));
    	matching = malloc((lenb + lena) * sizeof(char));
    /*}*/
    for (i = lenb - 1; i > xpos;)
    {
		stats[2]++;
		stats[4] += 1 - noEndGap;
		nbEndGap++;
/*		stats[0] += noEndGap;*/
		recon1[nbAligns] = '-';
		recon2[nbAligns] = b[i--];
		matching[nbAligns++] = (noEndGap ? 'e' : ' ');
		if (homoPol > 0)
			homopolHandling(_LEFT,b[i + 1],b[i + 1], homoPol, homopolState, matching, stats, noEndGap, nbAligns);
    }
    for (j = lena - 1; j > ypos;)
    {
		stats[2]++;
		stats[4] += 1 - noEndGap;
		nbEndGap++;
/*		stats[0] += noEndGap;*/
		recon1[nbAligns] = a[j--];
		recon2[nbAligns] = '-';
		matching[nbAligns++] = (noEndGap ? 'e' : ' ');
		if (homoPol > 0)
			homopolHandling(_UP,a[j + 1],a[j + 1], homoPol, homopolState, matching, stats, noEndGap, nbAligns);
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
				recon1[nbAligns] = a[ypos--];
				recon2[nbAligns] = b[xpos--];
				matching[nbAligns++] = (mp ? '|' : '.');
				if (homoPol > 0)
					homopolHandling(_DIAG,a[ypos + 1],b[xpos + 1], homoPol, homopolState, matching, stats, mp, nbAligns);
				break;
			case _LEFT:
				stats[2]++;
				/*stats[0] += (ypos == (lena - 1)	&& noEndGap ? 1 : 0);*/
				stats[4] += (ypos == (lena - 1)	&& noEndGap ? 0 : (lastGap && oneGap ? 0 : 1));
				if (ypos == (lena - 1))
					nbEndGap++;
				recon1[nbAligns] = '-';
				recon2[nbAligns] = b[xpos--];
				matching[nbAligns++] = (ypos == (lena - 1) && noEndGap ? 'e' : (lastGap && oneGap ? 'c' : ' '));
				lastGap = 1;
				if (homoPol > 0)
					homopolHandling(_LEFT,b[xpos + 1],b[xpos + 1], homoPol, homopolState, matching, stats, (lastGap && oneGap ? 0 : 1), nbAligns);
				break;
			case _UP:
				stats[2]++;
				/*stats[0] += (xpos == (lenb - 1) && noEndGap ? 1 : 0);*/
				stats[4] += (xpos == (lenb - 1) && noEndGap ? 0 : (lastGap && oneGap ? 0 : 1));
				if (xpos == (lenb - 1))
					nbEndGap++;
				recon1[nbAligns] = a[ypos--];
				recon2[nbAligns] = '-';
				matching[nbAligns++] = (xpos == (lenb - 1) && noEndGap ? 'e' : (lastGap && oneGap ? 'c' : ' '));
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
		stats[4] += 1 - noEndGap;
		nbEndGap++;
		/*stats[0] += noEndGap;*/
		recon1[nbAligns] = '-';
		recon2[nbAligns] = b[xpos];
		matching[nbAligns++] = (noEndGap ? 'e' : ' ');
		if (homoPol > 0)
			homopolHandling(_LEFT,b[xpos],b[xpos], homoPol, homopolState, matching, stats, 1 - noEndGap, nbAligns);
    }
    for (; ypos >= 0; ypos--)
    {
		stats[2]++;
		stats[4] += 1 - noEndGap;
		nbEndGap++;
		/*stats[0] += noEndGap;*/
		recon1[nbAligns] = a[ypos];
		recon2[nbAligns] = '-';
		matching[nbAligns++] = (noEndGap ? 'e' : ' ');
		if (homoPol > 0)
			homopolHandling(_UP,a[ypos],a[ypos], homoPol, homopolState, matching, stats, 1 - noEndGap, nbAligns);
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
	free(matching);
	free(recon1);
	free(recon2);
	free(start1);
	free(start2);
	if (!tstOut)
		return -1;
	if (dif)
		return stats[4];
	return 100.0f * stats[4] / nbAligns;
}

int				_needleAll(char *file1, char *file2, int nEG, float dTh, int hPo, int gOp, int gEx, int nOp, int nEx, char *mode, char *fOut, int dif, int oGa, int cut)
{
	int			len1, len2;
	int			curMalloc = 0;
	int			multiFile = 0;
	int			curComp = 0;
	int			curFirst = 0;
	int			verbose = 0;
	int			seqCount = 0;
	float		score = 0;
	FILE		*seqFile1;
	FILE		*seqFile2;
	FILE		*outHandle;
	char		*readBuffer;
	char		*seq1, *seq2, *seqID1, *seqID2;

	outHandle = fopen(fOut, "w+");
	fprintf(outHandle, "F&F Needleman report\n");
	fprintf(outHandle, "--------------------\n");
	fprintf(outHandle, " File1\t: %s\n", file1);
	fprintf(outHandle, " File2\t: %s\n", file1);
	fprintf(outHandle, " NoEndGap\t: %d\n", nEG);
	fprintf(outHandle, " DistThresh\t: %.2f\n", dTh);
	fprintf(outHandle, " Homopolymers\t: %d\n", hPo);
	fprintf(outHandle, " Gap Open\t: %d\n", gOp);
	fprintf(outHandle, " Gap Extend\t: %d\n", gEx);
	fprintf(outHandle, " End Open\t: %d\n", nOp);
	fprintf(outHandle, " End Extend\t: %d\n", nEx);
	fprintf(outHandle, " Out mode\t: %s\n", mode);
	fprintf(outHandle, "\n");
	if (mode[0] == 'p')
	{
		verbose = 1;
		fclose(outHandle);
	}
	if (mode[0] == 's')
	{
		verbose = 0;
	}
	if (mode[0] == 'l')
	{
		verbose = 2;
	}
	if (file2 == NULL || file2[0] == 0)
	{
		multiFile = 0;
		file2 = file1;
	}
	else
		multiFile = 1;
	printf("Starting F&F Optimized Needleman\n");
	printf("File : %s\n", file1);
	printf("File : %s\n", file2);
	if ((seqFile1 = fastaOpen(file1)) == NULL)
	{
		printf("Unable to open file : %s\n", file1);;
		return;
	}
	curMalloc = SEQSIZE * SEQSIZE;
	seq1 = malloc(BUFFSIZE * sizeof(char));
	seq2 = malloc(BUFFSIZE * sizeof(char));
	readBuffer = malloc(BUFFSIZE * sizeof(char));
	seqID1 = malloc(33 * sizeof(char));
	seqID2 = malloc(33 * sizeof(char));
    int *m = malloc(curMalloc * sizeof(int));
    int *horizGap_m = malloc(curMalloc * sizeof(int));
    int *vertGap_m = malloc(curMalloc * sizeof(int));
    int *trBack = malloc(curMalloc * sizeof(int));
	len1 = fastaIterate(seqFile1, seq1, seqID1, readBuffer);
	while (len1 > 0)
	{
		if (verbose == 0)
			fprintf(outHandle, "%s ", seqID1);
		if (curFirst != 0 && (curFirst == seqCount) && multiFile == 0)
		{
			if (verbose == 0)
				fprintf(outHandle, "\n");
			break;
		}
		if ((seqFile2 = fastaOpen(file2)) == NULL)
		{
			printf("Unable to open file : %s\n", file2);
			return;
		}
		if (multiFile == 0)
			fastaSkip(seqFile2, curFirst + 1, readBuffer);
		len2 = fastaIterate(seqFile2, seq2, seqID2, readBuffer);
		while (len2 > 0)
		{
			if (curFirst == 0)
				seqCount++;
			if (len1 * len2 > curMalloc)
			{
				m = realloc(m, len1 * len2 * sizeof(int));
				horizGap_m = realloc(horizGap_m, len1 * len2 * sizeof(int));
				vertGap_m = realloc(vertGap_m, len1 * len2 * sizeof(int));
				trBack = realloc(trBack, len1 * len2 * sizeof(int));
				curMalloc = len1 * len2;
			}
			score = _needleAffine(seq1, seq2, nEG, dTh, hPo, gOp, gEx, nOp, nEx, len1, len2, m, horizGap_m, vertGap_m, trBack, verbose, fOut, dif, seqID1, seqID2, oGa, cut);
			if (verbose == 0)
			{
				if (score >= 0)
				{
					if (dif)
						fprintf(outHandle, "%d ", (int)score);
					else
						fprintf(outHandle, "%.4f ", score);
				}
				else
					fprintf(outHandle, "X ");

			}
			if (verbose == 2 && score >= 0)
			{
				fprintf(outHandle, "%s\t%s\t", seqID1, seqID2);
				if (dif)
					fprintf(outHandle, "%d\n", (int)score);
				else
					fprintf(outHandle, "%.4f\n", score);
			}
			/*if computeMat == '' or computeMat[curComp] == 1:*/
			len2 = fastaIterate(seqFile2, seq2, seqID2, readBuffer);
			curComp += 1;
		}
		if (verbose == 0)
			fprintf(outHandle, "\n");
		fclose(seqFile2);
		len1 = fastaIterate(seqFile1, seq1, seqID1, readBuffer);
		curFirst += 1;
	}
	fclose(seqFile1);
	if (verbose == 0 || verbose == 2)
		fclose(outHandle);
	free(seq1);
	free(seq2);
	free(seqID1);
	free(seqID2);
	free(readBuffer);
	free(trBack);
    free(horizGap_m);
    free(vertGap_m);
    free(m);
	return curComp;
}

static PyObject	*needleman_chord(PyObject* self, PyObject* args)
{
    int 		*f1;
    int 		*f2;
	char		*mode;
	char		*fOut;
    int 		nEG;
    float 		dTh;
    int 		hPo;
    int 		gOp;
    int 		gEx;
    int 		nOp;
    int 		nEx;
    int 		dif;
	int			oGa;
	int			cut;

    if (!PyArg_ParseTuple(args, "iissifiiiiiiii", &f1, &f2, &mode, &fOut, &nEG, &dTh, &cut, &hPo, &gOp, &gEx, &nOp, &nEx, &dif, &oGa))
        return NULL;
    return Py_BuildValue("i", _needleAll(f1, f2, nEG, dTh, hPo, gOp, gEx, nOp, nEx, mode, fOut, dif, oGa, cut));
}

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

static PyMethodDef myextension_methods[] = {
    {"needleall", (PyCFunction)needleman, METH_VARARGS, "Calculate Needleman-Wunsch for a whole set"},
    {NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int myextension_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int myextension_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "needleall",
        NULL,
        sizeof(struct module_state),
        myextension_methods,
        NULL,
        myextension_traverse,
        myextension_clear,
        NULL
};

#define INITERROR return NULL

PyObject *
PyInit_needleall(void)

#else
#define INITERROR return

void
PyInit_needleall(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("needleall", myextension_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("myextension.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
