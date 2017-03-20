#!/usr/bin/env python
# -*- coding: utf8 -*-

# Main script for music generation

import hyperopt
import os
import random
import logging
import cPickle as pkl
import subprocess
import glob
import time
import re
# Build data
from build_data import build_data
# Clean Script
from clean_result_folder import clean

import train


N_HP_CONFIG = 1
LOCAL = True
# For Guillimin, write in the project space. Home is too small (10Gb VS 1Tb)
if LOCAL:
    RESULT_ROOT = os.getcwd() + '/../'
else:
    RESULT_ROOT = "/sb/project/ymd-084-aa/leo/"
SOURCE_DIR = os.getcwd()
DATA_DIR = "../Data"

############################################################
# Logging
############################################################
# log file
log_file_path = 'log/main_log'
# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=log_file_path,
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

############################################################
# Script parameters
############################################################
logging.info('########################################################')
logging.info('Script parameters')
# Parameters
script_param = {}
# Model
script_param["model_name"] = 'Lstm'
# paths
script_param["data_folder"] = DATA_DIR
# data
script_param['temporal_granularity'] = 'event_level' # event_level, frame_level
script_param['quantization'] = 100
script_param['unit_type'] = 'binary'
script_param["max_translation"] = 2
script_param['skip_sample'] = 1
# train
script_param["optimizer"] = 'rmsprop'
script_param["max_iter"] = 1
script_param["validation_order"] = 5
script_param["number_strips"] = 6
script_param["min_number_iteration"] = 30
script_param["time_limit"] = 11

result_folder = RESULT_ROOT + u'Results/' + script_param['temporal_granularity'] + '/' + script_param["unit_type"] + '/' +\
    'quantization_' + str(script_param['quantization']) + '/' + script_param['optimizer'] + '/' + script_param["model_name"]
# Check if the result folder exists
if not os.path.isdir(result_folder):
    os.makedirs(result_folder)
script_param['result_folder'] = result_folder
# Clean result folder
clean(result_folder)

for k,v in script_param.iteritems():
    logging.info(str(k) + " : " + str(v))

############################################################
# Check that the database we plan to use fit with our requirements
############################################################
metadata = pkl.load(open(DATA_DIR + '/metadata.pkl', 'rb'))
assert metadata["temporal_granularity"] == script_param['temporal_granularity']
assert metadata["quantization"] == script_param['quantization']
assert metadata["max_translation"] == script_param['max_translation']
assert metadata["unit_type"] == script_param['unit_type']

############################################################
# Load model
############################################################
if script_param["model_name"] == 'Lstm':
    from acidano.models.lop_keras.binary.lstm import Lstm as Model_class

############################################################
# Train hopt function
# Main thread, coordinate worker through a mongo db process
############################################################
logging.info((u'WITH HYPERPARAMETER OPTIMIZATION').encode('utf8'))
logging.info((u'**** Model : ' + script_param["model_name"]).encode('utf8'))
logging.info((u'**** Optimization technic : ' + script_param['optimizer']).encode('utf8'))
logging.info((u'**** Temporal granularity : ' + script_param['temporal_granularity']).encode('utf8'))
if script_param['unit_type'] == 'binary':
    logging.info((u'**** Binary unit (intensity discarded)').encode('utf8'))
elif script_param['unit_type'] == 'continuous':
    logging.info((u'**** Real valued unit (intensity taken into consideration)').encode('utf8'))
elif re.search('categorical', script_param['unit_type']):
    logging.info((u'**** Categorical units (discrete intensities)').encode('utf8'))
logging.info((u'**** Quantization : ' + str(script_param['quantization'])).encode('utf8'))
logging.info((u'**** Result folder : ' + str(script_param['result_folder'])).encode('utf8'))

############################################################
# Hyper parameter space
############################################################
model_space = Model_class.get_hp_space()

############################################################
# Grid search loop
############################################################
# Organisation :
# Each config is a folder with a random ID
# In eahc of this folder there is :
#    - a config.pkl file with the hyper-parameter space
#    - a result.txt file with the result
# The result.csv file containing id;result is created from the directory, rebuilt from time to time

# Already tested configs
list_config_folders = glob.glob(result_folder + '/*')

number_hp_config = max(0, N_HP_CONFIG - len(list_config_folders))
for hp_config in range(number_hp_config):
    # Give a random ID and create folder
    ID_SET = False
    while not ID_SET:
        ID_config = str(random.randint(0, 2**20))
        config_folder = script_param['result_folder'] + '/' + ID_config
        if not config_folder in list_config_folders:
            ID_SET = True
    os.mkdir(config_folder)

    # Find a point in space that has never been tested
    UNTESTED_POINT_FOUND = False
    while not UNTESTED_POINT_FOUND:
        model_space_config = hyperopt.pyll.stochastic.sample(model_space)
        space = {'model': model_space_config, 'script': script_param}
        # Check that this point in space has never been tested
        # By looking in all directories and reading the config.pkl file
        UNTESTED_POINT_FOUND = True
        for dirname in list_config_folders:
            this_config = pkl.load(open(dirname + '/config.pkl', 'rb'))
            if space == this_config:
                UNTESTED_POINT_FOUND = False
                break
    # Pickle the space in the config folder
    pkl.dump(space, open(config_folder + '/config.pkl', 'wb'))

    if LOCAL:
        import train_keras
        start_time_train = time.time()
        config_folder = config_folder
        params = pkl.load(open(config_folder + '/config.pkl', "rb"))
        train_keras.run_wrapper(params, config_folder, start_time_train)
    else:
        # Write pbs script
        file_pbs = config_folder + '/submit.pbs'
        text_pbs = """#!/bin/bash

#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l pmem=4000m
#PBS -l walltime=""" + str(train_param['walltime']) + """:00:00

module load iomkl/2015b Python/2.7.10 CUDA cuDNN
export OMPI_MCA_mtl=^psm

SRC=$HOME/lop/Source
cd $SRC
THEANO_FLAGS='device=gpu' python train_keras.py '""" + config_folder + "'"

        with open(file_pbs, 'wb') as f:
            f.write(text_pbs)

        # Launch script
        subprocess.call('qsub ' + file_pbs, shell=True)

    # Update folder list
    list_config_folders.append(config_folder)

# We done
# Processing results come afterward
