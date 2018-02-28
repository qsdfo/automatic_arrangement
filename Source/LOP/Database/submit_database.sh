#!/bin/bash

#PBS -N build_database
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=5:00:00

module load foss/2015b
module load Tensorflow/1.0.0-Python-2.7.12
source ~/Virtualenvs/lop/bin/activate

SRC=/home/crestel/automatic_arrangement/Source/LOP/Database
cd $SRC
python build_data_k_folds_pre_training.py