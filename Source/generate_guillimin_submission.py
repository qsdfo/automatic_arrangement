#!/usr/bin/env python
# -*- coding: utf8 -*-

import os


folder_path = 'script_folder'
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)
meta_script = folder_path + '/' + 'submission.sh'
f = open(meta_script, 'wb')
f.close()

def aux(algo, optim, gran, unit, quanti):
    pbs_content = """#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l pmem=4000m
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -q metaq
#PBS -M crestel.leopold@gmail.com
#PBS -m bae

module load iomkl/2015b Python/2.7.10 CUDA

SRC=$HOME/lop/Source
cd $SRC; python main.py """ +\
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
    with open(folder_path + '/' + fname, 'wb') as f:
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
