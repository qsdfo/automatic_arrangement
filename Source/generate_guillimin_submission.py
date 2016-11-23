#!/usr/bin/env python
# -*- coding: utf8 -*-

import os


meta_script = 'submission.sh'
f = open(meta_script, 'wb')
f.close()

def aux(algo, optim, gran, unit, quanti):
    pbs_content = """#!/bin/bash
    #
    # The line above tells Torque which shell PBS should use to run the
    # script (if you do not specify it, you could have surprises).
    # We can use the bash, sh, ksh, csh, tcsh, zsh, perl or python interpreters
    # although it is probably best to limit yourself to the classic ones
    # (first five).
    #

    # **************************************************************
    #
    #            TYPICAL PBS OPTION EXAMPLES
    #
    # **************************************************************

    # https://wiki.calculquebec.ca/w/Exemple_de_script_avec_documentation_des_options_de_qsub/en
    # To submit this file, use
    #          qsub example.pbs

    # **************************************************************
    # Load modules
    module load iomkl/2015b Python/2.7.10 CUDA
    # **************************************************************

    # **************************************************************
    # Account
    #PBS -A dpz-653-01

    # Ressources
    #PBS -l nodes=1:ppn=1:gpu=1
    #PBS -l pmem=4000m
    #PBS -l walltime=36:00:00

    # Log file
    #PBS -j oe

    # Queue
    #PBS -q aw

    # Sending of email:
    #PBS -M crestel.leopold@gmail.com
    #PBS -m bae

    # Interruption rules
    #PBS -r n
    # **************************************************************

    SRC=$HOME/lop/Source
    cd $SRC
    mpiexec -n 1 python main.py """ +\
        algo + " " +\
        optim + " " +\
        gran + " " +\
        unit + " " +\
        quanti

    # Write the pbs file for this configuration
    fname = algo + "_" +\
        optim + "_" +\
        gran + "_" +\
        unit + "_" +\
        quanti + ".pbs"
    with open(fname, 'wb') as f:
        f.write(pbs_content)

    # Add the qsub instruction in the main submission script
    with open(meta_script, 'ab') as f:
        f.write('qsub ' + fname + '\n')
    return

aux("cRBM", "gradient_descent", "event_level", "discrete_units", "4")
aux("cRnnRbm", "gradient_descent", "event_level", "discrete_units", "4")
aux("FGcRBM", "gradient_descent", "event_level", "discrete_units", "4")
aux("LSTM", "gradient_descent", "event_level", "discrete_units", "4")
aux("RBM", "gradient_descent", "event_level", "discrete_units", "4")
aux("RnnRbm", "gradient_descent", "event_level", "discrete_units", "4")

os.chmod(meta_script, 0755)
