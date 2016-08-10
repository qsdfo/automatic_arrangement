#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include "needleman.h"

#define _DIAG 	0
#define _UP  	1
#define _LEFT 	2

float 		_needlemanAbandonBUGDIST(char *seq1, char *seq2, int noEndGap, float distThresh, int homoPol, int gapOpen, int gapExtend, int diffs)
{
    int 	match = 1;
    int 	mismatch = -1;
    int 	gap = -1;
    int 	s1length;
    int		s2length;
    int		i, j;
    int 	matchScore;
    int 	upScore;
    int		leftScore;
    int 	nbAligns = 0;

    for (s1length = 0; seq1[s1length] != 0; s1length++);
    for (s2length = 0; seq2[s2length] != 0; s2length++);
    int **scores = (int **)malloc((s2length + 1) * sizeof(int *));
    int **scoreD = (int **)malloc((s2length + 1) * sizeof(int *));
    int **points = (int **)malloc((s2length + 1) * sizeof(int *));
    for (i = 0; i < s2length + 1; i++)
	{
		scores[i] = (int *)malloc((s1length + 1) * sizeof(int));
		scoreD[i] = (int *)malloc((s1length + 1) * sizeof(int));
		points[i] = (int *)malloc((s1length + 1) * sizeof(int));
		scores[i][0] = i * gap;
		scoreD[i][0] = (noEndGap == 0 ? i * gap : 0);
		points[i][0] = _UP;
	}
	for (i = 0; i < s1length + 1; i++)
	{
		scores[0][i] = i * gap;
		scoreD[0][i] = (noEndGap == 0 ? i * gap : 0);
		points[0][i] = _LEFT;
	}
    for (i = 1; i < s2length + 1; i++)
        for (j = 1; j < s1length + 1; j++)
        {
            matchScore = scores[i-1][j-1] + ((seq1[j-1] == seq2[i-1]) ? match : mismatch);
            upScore = scores[i-1][j] + (points[i-1][j] == _DIAG ? gapOpen : gapExtend);
            leftScore = scores[i][j-1] + (points[i][j-1] == _DIAG ? gapOpen : gapExtend);
            if ((matchScore >= leftScore) && (matchScore >= upScore))
            {
                points[i][j] = _DIAG;
                scores[i][j] = matchScore;
                scoreD[i][j] = scoreD[i-1][j-1] + (seq1[j-1] != seq2[i-1]);
            }
            else
            {
            	if (upScore >= leftScore)
            	{
            		points[i][j] = _UP;
            		scores[i][j] = upScore;
            		scoreD[i][j] = scoreD[i-1][j] + ((noEndGap == 0) || (i != s2length && j != s1length));
            	}
            	else
            	{
            		points[i][j] = _LEFT;
            		scores[i][j] = leftScore;
            		scoreD[i][j] = scoreD[i][j-1] + ((noEndGap == 0) || (i != s2length && j != s1length));
            	}
            }
        }
    if (diffs == 1)
        return scoreD[s2length][s1length];
    else
        return scoreD[s2length][s1length] * 100.0 / nbAligns;
}

float 		_needleman(char *seq1, char *seq2, int noEndGap, float distThresh, int homoPol, int gapOpen, int gapExtend, int diffs)
{
    int 	gap = -1;
    int 	s1length;
    int		s2length;
    int		i, j;
    int 	matchScore;
    int 	upScore;
    int		leftScore;
    int 	nbAligns = 0;
    int		verbose = 1;
    float 	dist = 0;
    float	addD = 0;
    char	*recon1, *recon2;
    FILE	*outputFile;
    int 	homopolState[6] = {'?', 0, 0, '?', 0, 0};

    for (s1length = 0; seq1[s1length] != 0; s1length++);
    for (s2length = 0; seq2[s2length] != 0; s2length++);
    int **scores = (int **)malloc((s1length + 1) * sizeof(int *));
    char **points = (char **)malloc((s1length + 1) * sizeof(char *));
    for (i = 0; i < s1length + 1; i++)
	{
		scores[i] = (int *)malloc((s2length + 1) * sizeof(int));
		points[i] = (char *)malloc((s2length + 1) * sizeof(char));
		scores[i][0] = gapOpen + ((i - 1) * gapExtend);
		points[i][0] = _UP;
	}
	for (i = 1; i < s2length + 1; i++)
	{
		scores[0][i] = gapOpen + ((i - 1) * gapExtend);
		points[0][i] = _LEFT;
	}
	scores[0][0] = 0;
    for (i = 1; i < s1length + 1; i++)
        for (j = 1; j < s2length + 1; j++)
        {
            matchScore = scores[i-1][j-1] + distance_chord(seq1[i-1], seq2[j-1]);
            upScore = scores[i-1][j] + (points[i-1][j] == _DIAG ? gapOpen : gapExtend);
            leftScore = scores[i][j-1] + (points[i][j-1] == _DIAG ? gapOpen : gapExtend);
            if ((matchScore >= leftScore) && (matchScore >= upScore))
            {
                points[i][j] = _DIAG;
                scores[i][j] = matchScore;
            }
            else
            {
            	if (leftScore >= upScore)
            	{
            		points[i][j] = _LEFT;
            		scores[i][j] = leftScore;
            	}
            	else
            	{
            		points[i][j] = _UP;
            		scores[i][j] = upScore;
            	}
            }
        }
    i = s1length;
    j = s2length;
    if (verbose != 0)
    {
    	recon1 = malloc((s1length + s2length) * sizeof(char));
    	recon2 = malloc((s1length + s2length) * sizeof(char));
    }
    while (i != 0 || j != 0)
	{
    	switch (points[i][j])
    	{
    	case _DIAG:
            dist += (seq1[i - 1] == seq2[j - 1] ? 0 : 1);
            if (verbose != 0)
            {
            	recon1[nbAligns] =	seq1[i - 1];
            	recon2[nbAligns] =	seq2[j - 1];
            }
            if (homoPol > 0)
            {
            	if (homopolState[0] != seq1[i - 1])
            	{
            		if (homopolState[1] >= homoPol && homopolState[4] >= homoPol)
            		{
            			dist -= homopolState[2];
            			homopolState[4] -= 10e5;
            		}
            		homopolState[0] = seq1[i - 1];
            		homopolState[1] = 1;
            		homopolState[2] = 0;
            	}
            	else
            	{
            		if (seq1[i - 1] == seq2[j - 1])
            			homopolState[1] += 1;
            		else
            		{
            			if (homopolState[1] >= homoPol && homopolState[4] >= homoPol)
            			{
            				dist -= homopolState[2];
            				homopolState[4] -= 10e5;
            			}
            			homopolState[1] -= 10e5;
            			homopolState[4] -= 10e5;
            		}
            	}
            	if (homopolState[3] != seq2[j - 1])
            	{
            		if (homopolState[4] >= homoPol && homopolState[1] >= homoPol)
            		{
            			dist -= homopolState[5];
            			homopolState[1] -= 10e5;
            		}
            		homopolState[3] = seq2[j - 1];
            		homopolState[4] = 1;
            		homopolState[5] = 0;
            	}
            	else
            	{
            		if (seq1[j-1] == seq2[j - 1])
            			homopolState[4] += 1;
            		else
            		{
            			if (homopolState[4] >= homoPol && homopolState[4] >= homoPol)
            			{
            				dist -= homopolState[5];
            				homopolState[1] -= 10e5;
            			}
            			homopolState[4] -= 10e5;
            			homopolState[1] -= 10e5;
            		}
            	}
            }
            free(points[i]);
        	free(scores[i]);
            i -= 1;
            j -= 1;
            break;
    	case _LEFT:
    		addD = (((noEndGap != 0) && ((i == 0 || j == 0) || (i == s2length || j == s1length))) ? 0 : 1);
    		dist += addD;
            if (verbose != 0)
            {
            	recon1[nbAligns] =	'-';
            	recon2[nbAligns] =	seq2[j - 1];
            }
    		if (homoPol > 0)
    		{
    			if (seq2[j-1] == homopolState[3])
    			{
    				homopolState[2] += addD;
    				homopolState[4] += 1;
    				homopolState[5] += addD;
    			}
    			else
    			{
    				if (homopolState[4] >= homoPol && homopolState[1] >= homoPol)
    				{
    					dist -= homopolState[5];
    					homopolState[1] -= 10e5;
    				}
    				homopolState[3] = seq2[j-1];
    				homopolState[4] = 1;
    				homopolState[5] = addD;
    				if (homopolState[1] >= homoPol && homopolState[4] >= homoPol)
    				{
    					dist -= homopolState[2];
    					homopolState[4] -= 10e5;
    				}
    				homopolState[0] = '-';
    				homopolState[1] = 0;
    				homopolState[2] = addD;
    			}
    		}
            j -= 1;
            break;
    	case _UP:
        	addD = (((noEndGap != 0) && ((i == 0 || j == 0) || (i == s2length || j == s1length))) ? 0 : 1);
        	dist += addD;
            if (verbose != 0)
            {
            	recon1[nbAligns] =	seq1[i - 1];
            	recon2[nbAligns] =	'-';
            }
        	if (homoPol > 0)
        	{
        		if (seq1[i - 1] == homopolState[0])
        		{
        			homopolState[1] += 1;
        			homopolState[2] += addD;
        			homopolState[5] += addD;
        		}
        		else
        		{
        			if (homopolState[1] >= homoPol && homopolState[4] >= homoPol)
        			{
        				dist -= homopolState[2];
        				homopolState[4] -= 10e5;
        			}
        			homopolState[0] = seq1[i - 1];
        			homopolState[1] = 1;
        			homopolState[2] = addD;
        			if (homopolState[4] >= homoPol && homopolState[1] >= homoPol)
        			{
        				dist -= homopolState[4];
        				homopolState[1] -= 10e5;
        			}
        			homopolState[3] = '-';
        			homopolState[4] = 0;
        			homopolState[5] = addD;
        		}
        	}
        	free(points[i]);
        	free(scores[i]);
            i -= 1;
            break;
    	default:
    		printf("[ERROR] : Unknown pointer in Needleman matrix\n");
    		return;
    	}
    	nbAligns = nbAligns + 1;
	}
    if (verbose != 0)
    {
    	outputFile = fopen("needleman_align.txt", "a+");
    	fprintf(outputFile, "**************************\n");
        fprintf(outputFile, "Length \t: %d\n", nbAligns);
        fprintf(outputFile, "Identities \t: %d (%f%%)\n", (int)(nbAligns - dist), (nbAligns - dist)/nbAligns);
    	fprintf(outputFile, "**************************\n");
    	for (j = nbAligns - 1; j >= 0; j -= 50)
    	{
    		for (i = j; i > j - 50 && i >= 0; i--)
    			fprintf(outputFile, "%c", recon1[i]);
    		fprintf(outputFile, "\n");
    		for (i = j; i > j - 50 && i >= 0; i--)
    			if (recon1[i] == recon2[i])
    				fprintf(outputFile, "|");
    			else
    				fprintf(outputFile, " ");
    		fprintf(outputFile, "\n");
    		for (i = j; i > j - 50 && i >= 0; i--)
    			fprintf(outputFile, "%c", recon2[i]);
    		fprintf(outputFile, "\n\n");
    	}
    	fclose(outputFile);
    	free(recon1);
    	free(recon2);
    }
    free(points);
    free(scores);
    if (diffs == 1)
        return dist;
    else
        return dist * 100.0 / nbAligns;
}

static PyObject	*needleman(PyObject* self, PyObject* args)
{
    char 		*s1;
    char 		*s2;
    int 		nEG;
    float 		dTh;
    int 		hPo;
    int 		gOp;
    int 		gEx;
    int 		dif;

    if (!PyArg_ParseTuple(args, "ssifiiii", &s1, &s2, &nEG, &dTh, &hPo, &gOp, &gEx, &dif))
        return NULL;
    return Py_BuildValue("f", _needleman(s1, s2, nEG, dTh, hPo, gOp, gEx, dif));
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
    {"needleman", (PyCFunction)needleman, METH_VARARGS, "Calculate Needleman-Wunsch global alignment"},
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
        "needleman",
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
PyInit_needleman(void)

#else
#define INITERROR return

void
PyInit_needleman(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("needleman", myextension_methods);
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
