'''
Needleman-Wunsch algorithm
@author: Philippe Esling
'''

import sys;

def homopolLen(seq, h):
    it = 0
    seqH = [];
    curHomopol = ['', 0]
    for char in seq:
        if curHomopol[0] != char:
            if curHomopol[1] >= h:
                seqH.extend([1 in xrange(curHomopol[1])])
            else:
                seqH.extend([0 in xrange(curHomopol[1])])
            curHomopol[0] = char
            curHomopol[1] = 1
        else:
            curHomopol[1] += 1
        it += 1
    if curHomopol[1] >= h:
        seqH.extend([1 in xrange(curHomopol[1])])
    else:
        seqH.extend([0 in xrange(curHomopol[1])])
    return it, seqH

def needlemanTwoPass(seq1, seq2, traceBack, noEndGap, distThresh, homoPol, gapOpen, gapExtend, diffs):
    match       = 1
    mismatch    = -1
    gap         = -1
    s1length = len(seq1)
    s2length = len(seq2)
    scores = [[(i * gap) + (j * gap) for i in xrange(s1length + 1)] for j in xrange(s2length + 1)]
    points = [['U' for i in range(s1length + 1)] for j in range(s2length + 1)]
    points[0][1:(s1length+1)] = s1length*'L'
    for i in xrange(1,s2length + 1):
        for j in xrange(1,s1length + 1):
            matchScore = scores[i-1][j-1] + (match if (seq1[j-1] == seq2[i-1]) else mismatch)
            upScore = scores[i-1][j] + (gapOpen if points[i-1][j] == 'D' else gapExtend)
            leftScore = scores[i][j-1] + (gapOpen if points[i][j-1] == 'D' else gapExtend)
            if matchScore >= leftScore and matchScore >= upScore:
                points[i][j] = 'D'
                scores[i][j] = matchScore
            elif upScore >= leftScore:
                points[i][j] = 'U'
                scores[i][j] = upScore
            else:
                points[i][j] = 'L'
                scores[i][j] = leftScore
    if traceBack != 0:
        align1 = ''
        align2 = ''
    i = s2length
    j = s1length
    dist = 0;
    homopolState = ['', 0, 0, '', 0, 0]
    while (i != 0 or j != 0):
        if points[i][j] == 'D':
            dist += (0 if seq1[j-1] == seq2[i-1] else 1)
            if homoPol > 0:
                if (homopolState[0] != seq1[j-1]):
                    if (homopolState[1] >= homoPol and homopolState[4] >= homoPol):
                        dist -= homopolState[2];
                        homopolState[4] -= 10e5
                    homopolState[0] = seq1[j-1]
                    homopolState[1] = 1
                    homopolState[2] = 0
                else:
                    if (seq1[j-1] == seq2[i-1]):
                        homopolState[1] += 1
                    else:
                        if (homopolState[1] >= homoPol and homopolState[4] >= homoPol):
                            dist -= homopolState[2];
                            homopolState[4] -= 10e5
                        homopolState[1] -= 10e5
                        homopolState[4] -= 10e5
                if (homopolState[3] != seq2[i-1]):
                    if (homopolState[4] >= homoPol and homopolState[1] >= homoPol):
                        dist -= homopolState[5];
                        homopolState[1] -= 10e5
                    homopolState[3] = seq2[i-1]
                    homopolState[4] = 1
                    homopolState[5] = 0
                else:
                    if (seq1[j-1] == seq2[i-1]):
                        homopolState[4] += 1
                    else:
                        if (homopolState[4] >= homoPol  and homopolState[4] >= homoPol):
                            dist -= homopolState[5];
                            homopolState[1] -= 10e5
                        homopolState[4] -= 10e5
                        homopolState[1] -= 10e5
            if traceBack != 0:
                align1 = ''.join((align1, seq1[j-1]))
                align2 = ''.join((align2, seq2[i-1]))
            i-=1
            j-=1
        elif points[i][j] == 'L':
            addD = (0 if ((noEndGap != 0) and ((i == 0 or j == 0) or (i == s2length or j == s1length))) else 1)
            dist += addD
            if homoPol > 0:
                if seq1[j-1] == homopolState[0]:
                    homopolState[1] += 1
                    homopolState[2] += addD
                    homopolState[5] += addD
                else:
                    if homopolState[1] >= homoPol and homopolState[4] >= homoPol:
                        dist -= homopolState[2]
                        homopolState[4] -= 10e5
                    homopolState[0] = seq1[j-1]
                    homopolState[1] = 1;
                    homopolState[2] = addD;
                    if homopolState[4] >= homoPol and homopolState[1] >= homoPol:
                        dist -= homopolState[4]
                        homopolState[1] -= 10e5
                    homopolState[3] = '-'
                    homopolState[4] = 0;
                    homopolState[5] = addD;
            if traceBack != 0:
                align1 = ''.join((align1, seq1[j-1]))
                align2 = ''.join((align2, '-'))
            j-=1
        else:
            addD = (0 if ((noEndGap != 0) and ((i == 0 or j == 0) or (i == s2length or j == s1length))) else 1)
            dist += addD
            if homoPol > 0:
                if seq2[i-1] == homopolState[3]:
                    homopolState[2] += addD
                    homopolState[4] += 1
                    homopolState[5] += addD
                else:
                    if homopolState[4] >= homoPol and homopolState[1] >= homoPol:
                        dist -= homopolState[5]
                        homopolState[1] -= 10e5
                    homopolState[3] = seq2[i-1]
                    homopolState[4] = 1;
                    homopolState[5] = addD;
                    if homopolState[1] >= homoPol and homopolState[4] >= homoPol:
                        dist -= homopolState[2]
                        homopolState[4] -= 10e5
                    homopolState[0] = '-'
                    homopolState[1] = 0;
                    homopolState[2] = addD;
            if traceBack != 0:
                align1 = ''.join((align1, '-'))
                align2 = ''.join((align2, seq2[i-1]))
            i-=1
    if traceBack != 0:
        print(align1[::-1])
        print(align2[::-1])
    if diffs:
        return dist
    else:
        return dist * 100.0 / float(max(s2length, s1length))

def needlemanOnePass(seq1, seq2, traceBack, noEndGap, distThresh, homoPol, gapOpen, gapExtend, diffs):
    match       = 1
    mismatch    = -1
    gap         = -1
    s1length = len(seq1)
    s2length = len(seq2)
    if noEndGap:
        scoreDist = [[0 for i in xrange(s1length + 1)] for j in xrange(s2length + 1)]
    else:
        scoreDist = [[i + j for i in xrange(s1length + 1)] for j in xrange(s2length + 1)]
    scores = [[(i * gap) + (j * gap) for i in xrange(s1length + 1)] for j in xrange(s2length + 1)]
    points = [['U' for i in range(s1length + 1)] for j in range(s2length + 1)]
    points[0][1:(s1length+1)] = s1length*'L'
    for i in xrange(1,s2length + 1):
        curRowMin = sys.maxint
        for j in xrange(1,s1length + 1):
            matchScore = scores[i-1][j-1] + (match if (seq1[j-1] == seq2[i-1]) else mismatch)
            # distance_chord(i-1,j-1)
            ##
            ##

            ##
            ##
            upScore = scores[i-1][j] + (gapOpen if points[i-1][j] == 'D' else gapExtend)
            leftScore = scores[i][j-1] + (gapOpen if points[i][j-1] == 'D' else gapExtend)
            if matchScore >= leftScore and matchScore >= upScore:
                points[i][j] = 'D'
                scores[i][j] = matchScore
                scoreDist[i][j] = scoreDist[i-1][j-1] + (seq1[j-1] != seq2[i-1])
            elif upScore >= leftScore:
                points[i][j] = 'U'
                scores[i][j] = upScore
                scoreDist[i][j] = scoreDist[i-1][j] + ((noEndGap == 0) or (i != s2length and j != s1length))
            else:
                points[i][j] = 'L'
                scores[i][j] = leftScore
                scoreDist[i][j] = scoreDist[i][j-1] + ((noEndGap == 0) or (i != s2length and j != s1length))
            curRowMin = curRowMin if scoreDist[i][j] > curRowMin else scoreDist[i][j]
        if curRowMin >= distThresh:
            print('* [Warning] : Abandon performed *')
            return curRowMin
    if traceBack:
        align1 = ''
        align2 = ''
        i = s2length
        j = s1length
        while (i != 0 or j != 0):
            if points[i][j] == 'D':
                align1 = ''.join((align1, seq1[j-1]))
                align2 = ''.join((align2, seq2[i-1]))
                i-=1
                j-=1
            elif points[i][j] == 'L':
                align1 = ''.join((align1, seq1[j-1]))
                align2 = ''.join((align2, '-'))
                j-=1
            else:
                align1 = ''.join((align1, '-'))
                align2 = ''.join((align2, seq2[i-1]))
                i-=1
        print(align1[::-1])
        print(align2[::-1])
    if diffs:
        return scoreDist[s2length][s1length], align1[::-1], align2[::-1]
    else:
        return scoreDist[s2length][s1length] * 100.0 / float(max(s2length, s1length)), align1[::-1], align2[::-1]

# Testing bag of patterns idea
#eDict = emptyPatterns('ACGT', 4)
#print(bagOfPatterns('GCAAAAAAAAACGCG', 4, eDict.copy()))
#print(bagOfPatterns('GCAACAACAAACGCG', 4, eDict.copy()))
#multipleBOP('geneticTest.fasta', '', '')

#profile.run("multipleDistances('uniprotVarsplice.fasta', '', ''); print");
#needlemanOnePass('GCAAAAAAACGCG', 'GCAAAATGCG',1,0,20,0,-2,-1,1);


if __name__ == '__main__':
    score = needlemanTwoPass(seq1='GCAGCA',
                                             seq2='TTTGCAGCATTTGCA',
                                             traceBack=1,
                                             noEndGap=0,
                                             distThresh=1e9,
                                             homoPol=0,
                                             gapOpen=-10,
                                             gapExtend=-1,
                                             diffs=0)
