import needleman_p;
import needleman;
import subprocess;
import profile;

#
# Comparing alignments to EMBOSS
#
def embossCalc(filin, GapOpen=10, GapExtend=3):
    filou=filin.replace('.fasta', '_needleall.txt')
    command=['../emboss/needleall', '-asequence', filin, '-bsequence', filin, '-gapopen', GapOpen, '-gapextend', GapExtend, '-outfile', filou, '-aformat3', 'score', '-verbose']
    print('Subprocessing needleman-wunsch Alignments of %s' % filin);
    ret=subprocess.Popen(command)
    ret.wait()
    return filou
    #"d" is the dico of the coding seqID to the coded seqID

#
# Needleman-Wunsch speed test
#
def needlemanSpeedTest(fName):
    curTest = 0
    seqFile = fasta.fOpen(fName, 'r')
    seqID1, seq1 = fasta.fIterate(seqFile)
    if seqID1 == 0:
        return
    seqID2, seq2 = fasta.fIterate(seqFile)
    while seqID2 != 0:
        #print(needleman_p.needlemanOnePass(seq1,seq2,0,0,8000,0,-2,-1,1))
        #print(needleman_p.needlemanTwoPass(seq1,seq2,0,0,8000,0,-2,-1,1))
        print(needleman.needleman(seq1,seq2,0,8000,0,-2,-1,1))
        seqID2, seq2 = fasta.fIterate(seqFile)
        curTest += 1
    #print(curTest)

def multipleDistances(file1, file2, computeMat):
    if file2 == '':
        multiFile = 0
        file2 = file1
    else:
        multiFile = 1;
    curComp = 0;
    curFirst = 1;
    seqFile1 = fasta.fOpen(file1, 'r+')
    seqID1, seq1 = fasta.fIterate(seqFile1)
    while seqID1 != 0:
        seqFile2 = fasta.fOpen(file2, 'r')
        if multiFile == 0:
            fasta.fSkip(seqFile2, curFirst)
        seqID2, seq2 = fasta.fIterate(seqFile2)
        while seqID2 != 0:
            if computeMat == '' or computeMat[curComp] == 1:
                print(needleman.needleman(seq1,seq2,0,0,500,0,-10,-3,1))
            seqID2, seq2 = fasta.fIterate(seqFile2)
            curComp += 1
        seqID1, seq1 = fasta.fIterate(seqFile1)
        curFirst += 1

profile.run("embossCalc('../datasets/geneticTest_XLight.fasta', '2', '2');print;");
#multipleDistances('../datasets/geneticTest_XLight.fasta', '../datasets/geneticTest_XLight.fasta', '');

#print(needleman.needleman('CGCCCAAAAATGC','CGCCCCCCAAAATGC',0,15,3,-2,-1,1));
#print(needleman.needleman('CSWCRYKMMBVHDNUCAAAAATGC','CGCUNDHVBCCCCCAAAATGC',0,15,3,-2,-1,1));
#
# Testing suite
#
'''
# Identical sequences
print("Identical sequences")
#print(needlemanOnePass('ACGT', 'ACGT',1,0,10,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('ACGT', 'ACGT',1,0,10,0,-2,-1,1));
#print(needlemanOnePass('ACCCCCGTCTGA', 'ACCCCCGTCTGA',1,0,10,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('ACCCCCGTCTGA', 'ACCCCCGTCTGA',1,0,10,0,-2,-1,1));
#print(needlemanOnePass('ACCCCCGTCTGGTCTGAAACCCCCGTCTGGTCTGAA', 'ACCCCCGTCTGGTCTGAAACCCCCGTCTGGTCTGAA',1,0,10,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('ACCCCCGTCTGGTCTGAAACCCCCGTCTGGTCTGAA', 'ACCCCCGTCTGGTCTGAAACCCCCGTCTGGTCTGAA',1,0,10,0,-2,-1,1));
# Substitution sequences
print("One substitution")
#print(needlemanOnePass('ACCT', 'ACGT',1,0,10,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('ACCT', 'ACGT',1,0,10,0,-2,-1,1));
print("Two substitutions")
#print(needlemanOnePass('ACCTTTTTACCT', 'ACGTTTNTACCT',1,0,10,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('ACCTTTTTACCT', 'ACGTTTNTACCT',1,0,10,0,-2,-1,1));
print("Multiple substitutions")
#print(needlemanOnePass('ACCCCCGTCTGGTCTGAAACCCCCGTCTGGTCTGAA', 'ACCCGCGTCTGGTCTAAAACCTCCGTCGGGTCTTGA',1,0,10,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('ACCCCCGTCTGGTCTGAAACCCCCGTCTGGTCTGAA', 'ACCCGCGTCTGGTCTAAAACCTCCGTCGGGTCTTGA',1,0,10,0,-2,-1,1));
# Insert sequences
print("One insert")
#print(needlemanOnePass('ACCGT', 'ACGT',1,0,10,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('ACCGT', 'ACGT',1,0,10,0,-2,-1,1));
print("Two inserts")
#print(needlemanOnePass('ACTCGT', 'ACGT',1,0,10,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('ACTCGT', 'ACGT',1,0,10,0,-2,-1,1));
print("Multiple inserts")
#print(needlemanOnePass('ACCCCCGTCTGGTCTGAAACCCCCGTCTGGTCTGAA', 'ACCACCCGTACTGTGTCTGAACACCCCACGTCTGTGTCATGAA',1,0,10,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('ACCCCCGTCTGGTCTGAAACCCCCGTCTGGTCTGAA', 'ACCACCCGTACTGTGTCTGAACACCCCACGTCTGTGTCATGAA',1,0,10,0,-2,-1,1));
# Gaps start
print("Start-end gaps count :")
#print(needlemanOnePass('CACGTC', 'ACGT',1,0,10,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('CACGTC', 'ACGT',1,0,10,0,-2,-1,1));
print("No end gaps (both) :")
#print(needlemanOnePass('CACGTC', 'ACGT',1,1,10,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('CACGTC', 'ACGT',1,1,10,0,-2,-1,1));
print("No end gaps (mul) :")
#print(needlemanOnePass('CACTCGTC', 'ACGT',1,1,10,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('CACTCGTC', 'ACGT',1,1,10,0,-2,-1,1));
print("Start-end gaps (complex) :")
#print(needlemanOnePass('AAAACCCCCGTCTGGTCTGAAACCCCCGTCTGGTCTGAAGGG', 'ACCACCCGTACTGTGTCTGAACACCCCACGTCTGTGTCATGAA',1,0,15,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('AAAACCCCCGTCTGGTCTGAAACCCCCGTCTGGTCTGAAGGG', 'ACCACCCGTACTGTGTCTGAACACCCCACGTCTGTGTCATGAA',1,0,15,0,-2,-1,1));
print("No end gaps (complex) :")
#print(needlemanOnePass('AAAACCCCCGTCTGGTCTGAAACCCCCGTCTGGTCTGAAGGG', 'ACCACCCGTACTGTGTCTGAACACCCCACGTCTGTGTCATGAA',1,1,15,0,-2,-1,1));
print(needleman_p.needlemanTwoPass('AAAACCCCCGTCTGGTCTGAAACCCCCGTCTGGTCTGAAGGG', 'ACCACCCGTACTGTGTCTGAACACCCCACGTCTGTGTCATGAA',1,1,15,0,-2,-1,1));
# Homo-polymers
print("Homopols (simple (h = 3))")
#print(needlemanOnePass('CAAAACGG', 'CAACGG',1,0,15,3,-2,-1,1));
print(needleman_p.needlemanTwoPass('CAAAACGG', 'CAACGG',1,0,15,3,-2,-1,1));
print(needleman.needleman('CAAAACGG', 'CAACGG',0,15,3,-2,-1,1));
print("Homopols (simple (h = 3))")
#print(needlemanOnePass('CAAAACGG', 'CAACGG',1,0,15,3,-2,-1,1));
print(needleman_p.needlemanTwoPass('CAAAAACGG', 'CAAACGG',1,0,15,3,-2,-1,1));
print(needleman.needleman('CAAAAACGG', 'CAAACGG',0,15,3,-2,-1,1));
print("Homopols (simple - reversed (h = 3))")
#print(needlemanOnePass('CAACGG', 'CAAAACGG',1,0,15,3,-2,-1,1));
print(needleman_p.needlemanTwoPass('CAACGG', 'CAAAACGG',1,0,15,3,-2,-1,1));
print(needleman.needleman('CAACGG', 'CAAAACGG',0,15,3,-2,-1,1));
print("Homopols (overlap (h = 2)")
#print(needlemanOnePass('CAAAACGG', 'CAACGG',1,0,15,2,-2,-1,1));
print(needleman_p.needlemanTwoPass('CAAACCGG', 'CAAACCCCGG',1,0,15,2,-2,-1,1));
print(needleman.needleman('CAAACCGG', 'CAAACCCCGG',0,15,2,-2,-1,1));
print("Homopols (complex (h = 3)")
print(needleman_p.needlemanTwoPass('CGCCAAAAATGC','CGCCCCCCAAAATGC',1,0,15,3,-2,-1,1));
print(needleman.needleman('CGCCAAAAATGC','CGCCCCCCAAAATGC',0,15,3,-2,-1,1));
print("Homopols (complex (h = 3)")
print(needleman_p.needlemanTwoPass('CGCCCAAAATGC','CGCCCCCCAAAATGC',1,0,15,3,-2,-1,1));
print(needleman.needleman('CGCCCAAAATGC','CGCCCCCCAAAATGC',0,15,3,-2,-1,1));
print("No homopols")
print(needleman_p.needlemanTwoPass('CGCCCAAAAATGC','CGCCCCCCAAAATGC',1,0,15,0,-2,-1,1));
print(needleman.needleman('CGCCCAAAAATGC','CGCCCCCCAAAATGC',0,15,0,-2,-1,1));
print("H=3")
print(needleman_p.needlemanTwoPass('CGCCCAAAAATGC','CGCCCCCCAAAATGC',1,0,15,3,-2,-1,1));
'''