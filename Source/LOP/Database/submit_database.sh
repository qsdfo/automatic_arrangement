#!/bin/bash

#PBS -j oe
#PBS -N job_outputs/build_database
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=5:00:00

module load foss/2015b
module load Tensorflow/1.0.0-Python-3.5.2
source ~/Virtualenvs/tf_3/bin/activate

SRC=/home/crestel/Source/automatic_arrangement/Source/LOP/Database
cd $SRC
python build_data_k_folds_pre_training.py