from distutils.core import setup, Extension
 
modNeedle = Extension('needleall', sources = ['../imports/fasta.c', 'needleall.c'])
 
setup (name = 'PackageName',
        version = '1.0',
        description = 'Optimization of the Needleman-Wunsch algorithm with affine gaps penalties for several sets of fasta files',
        ext_modules = [modNeedle])
