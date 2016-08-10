from distutils.core import setup, Extension
 
modNeedle = Extension('needleman_affine', sources = ['needleman_affine.c', '../imports/fasta.c'])
 
setup (name = 'PackageName',
        version = '1.0',
        description = 'Optimization of the Needleman-Wunsch algorithm with affine gaps penalties',
        ext_modules = [modNeedle])