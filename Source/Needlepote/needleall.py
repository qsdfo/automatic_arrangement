#from imports import fasta;
#from alignment import needleman;
import needleall;

'''
needleall(#File1, #File2, #Mode, #FileOut, #NoGapEnd, #DistThresh, #Cut, #Homopol, #GapOpen, #GapExtend, #EndOpen, #EndExtend, #Diffs, #oneGap)

#File1 : [REQUIRED]
First Fasta file to parse
#File2 : [OPT]
Second Fasta file to parse (if File2 == '' then all non-redundant pairwise computed from File1)
#Mode : [REQUIRED]
'pair' : export pairwise alignment file
'score' : export only (eventually triangular) score matrix
#NoGapEnd [0/1]
1 : Don't take gap ends into account in score (no diffs added)
0 : Take into account + add EndOpen and EndExtend to score matrix
#DistThresh
Print or not (function of diffs)
#Cut
Print Or not (function of end gaps)
#Homopol [0...N]
Check for homopolymers (0 = don't check / N = take N homopolymers into account)
#GapOpen, #GapExtend, #EndOpen, #EndExtend
Sets of penalties for algorithm
#Diffs
1 = Output nbDiffs
0 = Output %Diffs
#One Gap
1 = Contiguous Gaps = Only 1 diff
0 = All gaps = diffs
'''

needleall.needleall('../../datasets/Testing/geneticTest_XXLight.fasta', '', 'pair', 'FandF_NeedleOutP.txt', 1, 0, 100, 3, 10, 3, 0, 0, 0, 1)
#needleall.needleall('../../datasets/Testing/needleCheck_XLight.fasta', '', 'score', 'FandF_NeedleOutS.txt', 1, 90, 50, 3, 10, 3, 0, 0, 0, 1)
#needleall.needleall('../../datasets/Testing/needleCheck_XLight.fasta', '', 'list', 'FandF_NeedleOutL.txt', 1, 90, 50, 3, 10, 3, 0, 0, 0, 1)
#needleall.needleall('../datasets/geneticTest_XLight.fasta', '', 'pair', 'FandF_NeedleOut.txt', 1, 500, 3, 10, 3, 0, 0, 0)
