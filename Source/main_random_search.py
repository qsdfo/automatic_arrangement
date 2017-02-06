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
# Build data
from build_data import build_data
# Clean Script
from clean_result_folder import clean

####################
# Reminder for plotting tools
# import matplotlib.pyplot as plt
# Histogram
# n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# plt.show()

N_HP_CONFIG = 20
LOCAL = True
BUILD_DATABASE = True
DATABASE_PATH = '/home/aciditeam-leo/Aciditeam/database/Orchestration/Orchestration_checked'
# RESULT_ROOT = os.getcwd() + '/../'
# For Guillimin, write in the project space. Home is too small (10Gb VS 1Tb)
RESULT_ROOT = "/sb/project/ymd-084-aa/"

commands = [
    'LSTM',
    'gradient_descent',
    'event_level',
    'discrete_units',
    '4'
]

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
logging.info('Script paramaters')

# Store script parameters
script_param = {}

# Select a model (path to the .py file)
# Two things define a model : it's architecture and the optimization method
################### DISCRETE
script_param['model_class'] = commands[0]
if commands[0] == "RBM":
    from acidano.models.lop.discrete.RBM import RBM as Model_class
elif commands[0] == "cRBM":
    from acidano.models.lop.discrete.cRBM import cRBM as Model_class
elif commands[0] == "FGcRBM":
    from acidano.models.lop.discrete.FGcRBM import FGcRBM as Model_class
elif commands[0] == "LSTM":
    from acidano.models.lop.discrete.LSTM import LSTM as Model_class
elif commands[0] == "RnnRbm":
    from acidano.models.lop.discrete.RnnRbm import RnnRbm as Model_class
elif commands[0] == "cRnnRbm":
    from acidano.models.lop.discrete.cRnnRbm import cRnnRbm as Model_class
###################  REAL
elif commands[0] == "LSTM_gaussian_mixture":
    from acidano.models.lop.real.LSTM_gaussian_mixture import LSTM_gaussian_mixture as Model_class
elif commands[0] == "LSTM_gaussian_mixture_2":
    from acidano.models.lop.real.LSTM_gaussian_mixture_2 import LSTM_gaussian_mixture_2 as Model_class
###################
else:
    raise ValueError(commands[0] + " is not a model")

# Optimization
script_param['optimization_method'] = commands[1]
if commands[1] == "gradient_descent":
    from acidano.utils.optim import gradient_descent as Optimization_method
elif commands[1] == 'adam_L2':
    from acidano.utils.optim import adam_L2 as Optimization_method
elif commands[1] == 'rmsprop':
    from acidano.utils.optim import rmsprop as Optimization_method
elif commands[1] == 'sgd_nesterov':
    from acidano.utils.optim import sgd_nesterov as Optimization_method
else:
    raise ValueError(commands[1] + " is not an optimization method")

# Temporal granularity
script_param['temporal_granularity'] = commands[2]
if script_param['temporal_granularity'] not in ['frame_level', 'event_level']:
    raise ValueError(commands[2] + " is not temporal_granularity")

# Unit type
unit_type = commands[3]
if unit_type == 'continuous_units':
    script_param['binary_unit'] = False
elif unit_type == 'discrete_units':
    script_param['binary_unit'] = True
else:
    raise ValueError("Wrong units type")

# Quantization
try:
    script_param['quantization'] = int(commands[4])
except ValueError:
    print(commands[4] + ' is not an integer')
    raise

############################################################
# System paths
############################################################
logging.info('System paths')
SOURCE_DIR = os.getcwd()
result_folder = RESULT_ROOT + u'Results/' + script_param['temporal_granularity'] + '/' + unit_type + '/' +\
    'quantization_' + str(script_param['quantization']) + '/' + Optimization_method.name() + '/' + Model_class.name()

# Check if the result folder exists
if not os.path.isdir(result_folder):
    os.makedirs(result_folder)
script_param['result_folder'] = result_folder
# Clean result folder
clean(result_folder)

# Data : .pkl files
data_folder = SOURCE_DIR + '/../Data'
if not os.path.isdir(data_folder):
    os.mkdir(data_folder)
script_param['data_folder'] = data_folder
script_param['skip_sample'] = 1

############################################################
# Train parameters
############################################################
train_param = {}
# Fixed hyper parameter
train_param['max_iter'] = 200        # nb max of iterations when training 1 configuration of hparams
# Config is set now, no need to modify source below for standard use

# Validation
train_param['validation_order'] = 5
train_param['initial_derivative_length'] = 20
train_param['check_derivative_length'] = 5

# Now, we can log to the root logger, or any other logger. First the root...
logging.info('#'*40)
logging.info('#'*40)
logging.info('#'*40)
logging.info('* L * O * P *')

############################################################
# Train hopt function
# Main thread, coordinate worker through a mongo db process
############################################################
logging.info((u'WITH HYPERPARAMETER OPTIMIZATION').encode('utf8'))
logging.info((u'**** Model : ' + Model_class.name()).encode('utf8'))
logging.info((u'**** Optimization technic : ' + Optimization_method.name()).encode('utf8'))
logging.info((u'**** Temporal granularity : ' + script_param['temporal_granularity']).encode('utf8'))
if script_param['binary_unit']:
    logging.info((u'**** Binary unit (intensity discarded)').encode('utf8'))
else:
    logging.info((u'**** Real valued unit (intensity taken into consideration)').encode('utf8'))
logging.info((u'**** Quantization : ' + str(script_param['quantization'])).encode('utf8'))
logging.info((u'**** Result folder : ' + str(script_param['result_folder'])).encode('utf8'))

############################################################
# Build data
############################################################
if BUILD_DATABASE:
    logging.info('# ** BUILD DATABASE **')
    index_files_dict = {}
    index_files_dict['train'] = [
        # DATABASE_PATH + "/debug_train.txt",
        DATABASE_PATH + "/bouliane_train.txt",
        DATABASE_PATH + "/hand_picked_Spotify_train.txt",
        DATABASE_PATH + "/liszt_classical_archives_train.txt"
    ]
    index_files_dict['valid'] = [
        # DATABASE_PATH + "/debug_valid.txt",
        DATABASE_PATH + "/bouliane_valid.txt",
        DATABASE_PATH + "/hand_picked_Spotify_valid.txt",
        DATABASE_PATH + "/liszt_classical_archives_valid.txt"
    ]
    index_files_dict['test'] = [
        # DATABASE_PATH + "/debug_test.txt",
        DATABASE_PATH + "/bouliane_test.txt",
        DATABASE_PATH + "/hand_picked_Spotify_test.txt",
        DATABASE_PATH + "/liszt_classical_archives_test.txt"
    ]

    build_data(root_dir=DATABASE_PATH,
               index_files_dict=index_files_dict,
               meta_info_path=data_folder + '/temp.p',
               quantization=script_param['quantization'],
               temporal_granularity=script_param['temporal_granularity'],
               store_folder=data_folder,
               logging=logging)

############################################################
# Hyper parameter space
############################################################
model_space = Model_class.get_hp_space()
optim_space = Optimization_method.get_hp_space()

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
        ID_config = str(random.randint(0, 2**25))
        config_folder = script_param['result_folder'] + '/' + ID_config
        if not config_folder in list_config_folders:
            ID_SET = True
    os.mkdir(config_folder)

    # Find a point in space that has never been tested
    UNTESTED_POINT_FOUND = False
    while not UNTESTED_POINT_FOUND:
        model_space_config = hyperopt.pyll.stochastic.sample(model_space)
        optim_space_config = hyperopt.pyll.stochastic.sample(optim_space)
        space = {'model': model_space_config, 'optim': optim_space_config, 'train': train_param, 'script': script_param}
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
        process = subprocess.Popen("THEANO_FLAGS='device=gpu1' python train.py '" + config_folder + "'", shell=True, stdout=subprocess.PIPE)
        process.wait()
    else:
        # Write pbs script
        file_pbs = config_folder + '/submit.pbs'
        text_pbs = """#!/bin/bash

#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l pmem=4000m
#PBS -l walltime=10:00:00

module load iomkl/2015b Python/2.7.10 CUDA cuDNN
OMPI_MCA_mtl=^psm

SRC=$HOME/lop/Source
cd $SRC
THEANO_FLAGS='device=gpu' python train.py '""" + config_folder + "'"

        with open(file_pbs, 'wb') as f:
            f.write(text_pbs)

        # Launch script
        subprocess.call('qsub ' + file_pbs, shell=True)

    # Update folder list
    list_config_folders.append(config_folder)

# We done
# Processing results come afterward
