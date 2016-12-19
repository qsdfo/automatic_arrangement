#!/usr/bin/env python
# -*- coding: utf8 -*-

# Main script for music generation

import functools
import os
import logging
import sys
import cPickle as pkl
import subprocess
# Hyperopt
from hyperopt import fmin, tpe
from hyperopt.mongoexp import MongoTrials
# Build data
from build_data import build_data
# Run wrapper
from run import run_wrapper

####################
# Reminder for plotting tools
# import matplotlib.pyplot as plt
# Histogram
# n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# plt.show()

############################################################
# Logging
############################################################
# log file
log_file_path = 'main_log'
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

REBUILD_DATABASE = False

############################################################
# Script parameters
############################################################
logging.info('Script paramaters')
script_param = {}
# Build data parameters :
# Temporal granularity
if len(sys.argv) < 4:
    script_param['temporal_granularity'] = u'event_level'
else:
    script_param['temporal_granularity'] = sys.argv[3]
    if script_param['temporal_granularity'] not in ['frame_level', 'event_level']:
        raise ValueError(sys.argv[3] + " is not temporal_granularity")

# Unit type
if len(sys.argv) < 5:
    script_param['binary_unit'] = True
else:
    unit_type = sys.argv[4]
    if unit_type == 'continuous_units':
        script_param['binary_unit'] = False
    elif unit_type == 'discrete_units':
        script_param['binary_unit'] = True
    else:
        raise ValueError("Wrong units type")

# Quantization
if len(sys.argv) < 6:
    script_param['quantization'] = 4
else:
    try:
        script_param['quantization'] = int(sys.argv[5])
    except ValueError:
        print(sys.argv[5] + ' is not an integer')
        raise

# Select a model (path to the .py file)
# Two things define a model : it's architecture and the optimization method
################### DISCRETE
script_param['model_class'] = sys.argv[1]
if sys.argv[1] == "RBM":
    from acidano.models.lop.discrete.RBM import RBM as Model_class
    if not script_param['binary_unit']:
        logging.warning("You're using a model defined for binary units with real valued units")
elif sys.argv[1] == "cRBM":
    from acidano.models.lop.discrete.cRBM import cRBM as Model_class
    if not script_param['binary_unit']:
        logging.warning("You're using a model defined for binary units with real valued units")
elif sys.argv[1] == "FGcRBM":
    from acidano.models.lop.discrete.FGcRBM import FGcRBM as Model_class
    if not script_param['binary_unit']:
        logging.warning("You're using a model defined for binary units with real valued units")
elif sys.argv[1] == "LSTM":
    from acidano.models.lop.discrete.LSTM import LSTM as Model_class
    if not script_param['binary_unit']:
        logging.warning("You're using a model defined for binary units with real valued units")
elif sys.argv[1] == "RnnRbm":
    from acidano.models.lop.discrete.RnnRbm import RnnRbm as Model_class
    if not script_param['binary_unit']:
        logging.warning("You're using a model defined for binary units with real valued units")
elif sys.argv[1] == "cRnnRbm":
    from acidano.models.lop.discrete.cRnnRbm import cRnnRbm as Model_class
    if not script_param['binary_unit']:
        logging.warning("You're using a model defined for binary units with real valued units")
###################  REAL
elif sys.argv[1] == "LSTM_gaussian_mixture":
    from acidano.models.lop.real.LSTM_gaussian_mixture import LSTM_gaussian_mixture as Model_class
    if script_param['binary_unit']:
        logging.warning("You're using a model defined for real valued units with binary units")
elif sys.argv[1] == "LSTM_gaussian_mixture_2":
    from acidano.models.lop.real.LSTM_gaussian_mixture_2 import LSTM_gaussian_mixture_2 as Model_class
    if script_param['binary_unit']:
        logging.warning("You're using a model defined for real valued units with binary units")
###################
else:
    raise ValueError(sys.argv[1] + " is not a model")

script_param['optimization_method'] = sys.argv[2]
if sys.argv[2] == "gradient_descent":
    from acidano.utils.optim import gradient_descent as Optimization_method
elif sys.argv[2] == 'adam_L2':
    from acidano.utils.optim import adam_L2 as Optimization_method
elif sys.argv[2] == 'rmsprop':
    from acidano.utils.optim import rmsprop as Optimization_method
elif sys.argv[2] == 'sgd_nesterov':
    from acidano.utils.optim import sgd_nesterov as Optimization_method
else:
    raise ValueError(sys.argv[2] + " is not an optimization method")

############################################################
# System paths
############################################################
logging.info('System paths')
SOURCE_DIR = os.getcwd()
result_folder = SOURCE_DIR + u'/../Results/' + unit_type + '/' + script_param['temporal_granularity'] + '/' +\
    'quantization_' + str(script_param['quantization']) + '/' + Optimization_method.name() + '/' + Model_class.name()

# Check if the result folder exists
if not os.path.isdir(result_folder):
    os.makedirs(result_folder)

# Count number of files in the folder and name this hparam run by the number
hparam_run_counter = 0
while(os.path.isdir(result_folder + '/hrun_' + str(hparam_run_counter))):
        hparam_run_counter += 1
result_folder = result_folder + '/hrun_' + str(hparam_run_counter)
os.mkdir(result_folder)
script_param['result_folder'] = result_folder

# Data : .pkl files
data_folder = SOURCE_DIR + '/../Data'
if not os.path.isdir(data_folder):
    os.mkdir(data_folder)
script_param['data_folder'] = data_folder

############################################################
# Train parameters
############################################################
train_param = {}
# Fixed hyper parameter
max_evals = 10000                    # number of hyper-parameter configurations evaluated
train_param['max_iter'] = 100        # nb max of iterations when training 1 configuration of hparams
# Config is set now, no need to modify source below for standard use

# Validation
train_param['validation_order'] = 5
train_param['initial_derivative_length'] = 20
train_param['check_derivative_length'] = 5

# Generation
train_param['generation_length'] = 50
train_param['seed_size'] = 10
train_param['quantization_write'] = script_param['quantization']

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
if REBUILD_DATABASE:
    logging.info('# ** Database REBUILT **')
    PREFIX_INDEX_FOLDER = SOURCE_DIR + "/../Data/Index/"
    index_files_dict = {}
    index_files_dict['train'] = [
        PREFIX_INDEX_FOLDER + "debug_train.txt",
        # PREFIX_INDEX_FOLDER + "bouliane_train.txt",
        # PREFIX_INDEX_FOLDER + "hand_picked_Spotify_train.txt",
        # PREFIX_INDEX_FOLDER + "liszt_classical_archives_train.txt"
    ]
    index_files_dict['valid'] = [
        PREFIX_INDEX_FOLDER + "debug_valid.txt",
        # PREFIX_INDEX_FOLDER + "bouliane_valid.txt",
        # PREFIX_INDEX_FOLDER + "hand_picked_Spotify_valid.txt",
        # PREFIX_INDEX_FOLDER + "liszt_classical_archives_valid.txt"
    ]
    index_files_dict['test'] = [
        PREFIX_INDEX_FOLDER + "debug_test.txt",
        # PREFIX_INDEX_FOLDER + "bouliane_test.txt",
        # PREFIX_INDEX_FOLDER + "hand_picked_Spotify_test.txt",
        # PREFIX_INDEX_FOLDER + "liszt_classical_archives_test.txt"
    ]

    build_data(index_files_dict=index_files_dict,
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
space = {'model': model_space, 'optim': optim_space, 'train': train_param, 'script': script_param}

############################################################
# MongoDB
############################################################
host = "localhost"
port = 27017
db_name =\
    unit_type[0] + '-' +\
    script_param['temporal_granularity'][0] + '-' +\
    str(script_param['quantization']) + '-' +\
    Optimization_method.name() + '-' +\
    Model_class.name()
mongo_adress = 'mongo://' + host + ':' + str(port) + '/' + db_name

# Stop a potential mongo worker
subprocess.call("ps -ef | grep mongod | grep -v grep | wc -l | tr -d ' ' > check_mongo_process_running.txt", shell=True)
with open('check_mongo_process_running.txt', 'rb') as f:
    number_mongodb_runnning = int(f.readline())
if number_mongodb_runnning > 0:
    subprocess.call("mongo admin --eval 'db.shutdownServer()'", shell=True)

# Start a mongodb process
logging.info("Start a mongo db process at " + mongo_adress)
if not os.path.isdir(db_name):
    os.mkdir(db_name)

mongo_str = "mongod --dbpath " + db_name + " --port " + str(port) + " --directoryperdb --fork --journal --logpath log.log --nohttpinterface"
# mongo_str = "numactl --interleave=all mongod --dbpath " + db_name + " --port " + str(port) + " --directoryperdb --fork --journal --logpath log.log --nohttpinterface"

subprocess.call(mongo_str, shell=True)

# Produce submission file (called N_worker time with qsub)
qsub_fname = 'submit_worker.pbs'
qsub_content = """#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l pmem=4000m
#PBS -l walltime=12:00:00
#PBS -q metaq

module load iomkl/2015b Python/2.7.10 CUDA

SRC=$HOME/lop/Source
cd $SRC
python hyperopt-mongo-worker.py --mongo=""" + host + ':' + str(port) + '/' + db_name +\
    """ --poll-interval=5""" + \
    """ --max-jobs=1""" + \
    """ --workdir=$SRC/""" + db_name

with open(qsub_fname, 'wb') as f:
    f.write(qsub_content)

# Create MongoTrials
trials = MongoTrials(mongo_adress + '/jobs')
tpe_suggest=functools.partial(
    tpe.suggest,
    n_startup_jobs=25,  # -- number of random jobs before optimization
)

############################################################
# Run : here this function will stop execution,
# and simply wait for mongo workers to connect to it,
# ask for jobs, and gather their results in the database
############################################################
logging.info('fmin call')
best = fmin(run_wrapper, space, trials=trials, algo=tpe_suggest, max_evals=max_evals)

############################################################
# Save the trials.results dictionnaries
############################################################
trials_file = script_param['result_folder'] + '/' + 'trials.pkl'
with open(trials_file, 'wb') as f:
    pkl.dump(trials, f, protocol=pkl.HIGHEST_PROTOCOL)
