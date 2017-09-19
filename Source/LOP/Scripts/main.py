#!/usr/bin/env python
# -*- coding: utf8 -*-

import cPickle as pkl
import re
import logging
import glob
import os
import shutil
import time
import numpy as np

from train import train
from LOP.Database.load_data import load_data_train, load_data_valid, load_data_test

# MODEL
from LOP.Models.LSTM_plugged_base import LSTM_plugged_base as Model
# DATABASE
DATABASE = "Data__event_level100__0"
DATABASE_PATH = "../../../Data/" + DATABASE
# HYPERPARAM ?
DEFINED_CONFIG = True
# RESULTS
result_folder =  '../../../Results/' + DATABASE + '/' + Model.name()
if not os.path.isdir(result_folder):
	os.makedirs(result_folder)

# Parameters
parameters = {
	"result_folder": result_folder,
	# Data
	"binarize_piano": True,
	"binarize_orchestra": True,
	"skip_sample": 1,
	"avoid_silence": True,
	# Train
	"max_iter": 3,            # nb max of iterations when training 1 configuration of hparams
	"walltime": 11,             # in hours
	# Validation
	"validation_order": 2,
	"number_strips": 3,
	# Hyperopt
	"random_search": False,     # If false, use a defined configuration
	"max_evals": 50,            # number of hyper-parameter configurations evaluated
}

# Load the database metadata and add them to the script parameters to keep a record of the data processing pipeline
parameters.update(pkl.load(open(DATABASE_PATH + '/metadata.pkl', 'rb')))


############################################################
# Logging
############################################################
# log file
log_file_path = 'log'
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

# Now, we can log to the root logger, or any other logger. First the root...
logging.info('#'*40)
logging.info('#'*40)
logging.info('#'*40)
logging.info('* L * O * P *')
logging.info((u'**** Model : ' + Model.name()).encode('utf8'))
logging.info((u'**** Optimization technic : ##########').encode('utf8'))
logging.info((u'**** Temporal granularity : ' + parameters['temporal_granularity']).encode('utf8'))
logging.info((u'**** Quantization : ' + str(parameters['quantization'])).encode('utf8'))
logging.info((u'**** Binary piano : ' + str(parameters["binarize_piano"])).encode('utf8'))
logging.info((u'**** Binary orchestra : ' + str(parameters["binarize_orchestra"])).encode('utf8'))
logging.info((u'**** Result folder : ' + result_folder).encode('utf8'))

############################################################
# Hyper parameter space
############################################################
model_parameters = Model.get_hp_space()

track_paths = [
		'/home/aciditeam-leo/Aciditeam/database/Orchestration/LOP_database_30_06_17/liszt_classical_archives/16',
		'/home/aciditeam-leo/Aciditeam/database/Orchestration/LOP_database_30_06_17/liszt_classical_archives/17',
		'/home/aciditeam-leo/Aciditeam/database/Orchestration/LOP_database_30_06_17/liszt_classical_archives/26',
		'/home/aciditeam-leo/Aciditeam/database/Orchestration/LOP_database_30_06_17/liszt_classical_archives/5',
]

configs = {
	'0': {
		'batch_size' : 200,
		'temporal_order' : 5,
		'dropout_probability' : 0,
		'weight_decay_coeff' : 0,
		'n_hidden' : [500, 500],
		'threshold' : 0,
		'weighted_ce' : 0
	},
}

def run_wrapper(parameters, model_params, config_folder, data_folder):

	############################################################
	# Load data
	############################################################
	time_load_0 = time.time()
	piano_train, orchestra_train, train_index \
		= load_data_train(data_folder,
						  model_params['temporal_order'],
						  model_params['batch_size'],
						  skip_sample=parameters['skip_sample'],
						  avoid_silence=parameters['avoid_silence'],
						  binarize_piano=parameters['binarize_piano'],
						  binarize_orchestra=parameters['binarize_orchestra'],
						  logger_load=logging)
	piano_valid, orchestra_valid, valid_index \
		= load_data_valid(data_folder,
						  model_params['temporal_order'],
						  model_params['batch_size'],
						  skip_sample=parameters['skip_sample'],
						  avoid_silence=True,
						  binarize_piano=parameters['binarize_piano'],
						  binarize_orchestra=parameters['binarize_orchestra'],
						  logger_load=logging)
	# This load is only for sanity check purposes
	piano_test, orchestra_test, _, _ \
		= load_data_test(data_folder,
						 model_params['temporal_order'],
						 model_params['batch_size'],
						 skip_sample=parameters['skip_sample'],
						 avoid_silence=True,
						 binarize_piano=parameters['binarize_piano'],
						 binarize_orchestra=parameters['binarize_orchestra'],
						 logger_load=logging)
	time_load_1 = time.time()
	logging.info('TTT : Loading data took {} seconds'.format(time_load_1-time_load_0))

	############################################################
	# Get dimensions of batches
	############################################################
	piano_dim = piano_train.shape[1]
	orchestra_dim = orchestra_train.shape[1]
	dimensions = {'batch_size': model_params['batch_size'],
				  'temporal_order': model_params['temporal_order'],
				  'piano_dim': piano_dim,
				  'orchestra_dim': orchestra_dim}

	############################################################
	# Update train_param and model_param dicts with new information from load data
	############################################################
	n_train_batches = len(train_index)
	n_val_batches = len(valid_index)

	logging.info((u'##### Data').encode('utf8'))
	logging.info((u'# n_train_batch :  {}'.format(n_train_batches)).encode('utf8'))
	logging.info((u'# n_val_batch :  {}'.format(n_val_batches)).encode('utf8'))

	parameters['n_train_batches'] = n_train_batches
	parameters['n_val_batches'] = n_val_batches

	# Class normalization
	notes_activation = orchestra_train.sum(axis=0)
	notes_activation_norm = notes_activation.mean() / (notes_activation+1e-10)
	class_normalization = np.maximum(1, np.minimum(20, notes_activation_norm))
	model_params['class_normalization'] = class_normalization

	# Other kind of regularization
	L_train = orchestra_train.shape[0]
	mean_notes_activation = notes_activation / L_train
	mean_notes_activation = np.where(mean_notes_activation == 0, 1. / L_train, mean_notes_activation)
	model_params['mean_notes_activation'] = mean_notes_activation

	############################################################
	# Instanciate model and Optimization method
	############################################################
	# logging.info((u'##### Model parameters').encode('utf8'))
	# for k, v in model_param.iteritems():
	#     logging.info((u'# ' + k + ' :  {}'.format(v)).encode('utf8'))
	# logging.info((u'##### Optimization parameters').encode('utf8'))
	# for k, v in optim_param.iteritems():
	#     logging.info((u'# ' + k + ' :  {}'.format(v)).encode('utf8'))

	model = Model(model_params, dimensions)

	############################################################
	# Train
	############################################################
	time_train_0 = time.time()
	loss, accuracy, best_epoch, best_model = train(model,
									piano_train, orchestra_train, train_index,
									piano_valid, orchestra_valid, valid_index,
									parameters, config_folder, time_train_0, logging)

	time_train_1 = time.time()
	training_time = time_train_1-time_train_0
	logging.info('TTT : Training data took {} seconds'.format(training_time))
	logging.info((u'# Best model obtained at epoch :  {}'.format(best_epoch)).encode('utf8'))
	logging.info((u'# Accuracy :  {}'.format(accuracy)).encode('utf8'))
	logging.info((u'###################\n').encode('utf8'))

	############################################################
	# Pickle (save) the model for plotting weights
	############################################################
	save_model_file = config_folder + '/model.pkl'
	with open(save_model_file, 'wb') as f:
		pkl.dump(best_model, f, protocol=pkl.HIGHEST_PROTOCOL)

	############################################################
	# Write result in a txt file
	############################################################
	result_file_path = config_folder + '/result.csv'
	with open(result_file_path, 'wb') as f:
		f.write("accuracy;" + str(accuracy) + '\n' + "loss;" + str(loss))

	# Close handler
	logging.removeHandler(hdlr)
	return

############################################################
# Grid search loop
############################################################
# Organisation :
# Each config is a folder with a random ID (integer)
# In eahc of this folder there is :
#    - a config.pkl file with the hyper-parameter space
#    - a result.txt file with the result
# The result.csv file containing id;result is created from the directory, rebuilt from time to time

if DEFINED_CONFIG:
	for config_id, model_parameters in configs.iteritems():
		config_folder = parameters['result_folder'] + '/' + config_id
		if not os.path.isdir(config_folder):
			os.mkdir(config_folder)
		else:
			user_input = raw_input("This folder already exists. Type y to overwrite : ")
			if user_input == 'y':
				shutil.rmtree(config_folder)
				os.mkdir(config_folder)	
			else:
				raise Exception("Config not overwritten")
		
		# Pickle the space in the config folder
		pkl.dump(model_parameters, open(config_folder + '/model_parameters.pkl', 'wb'))
		pkl.dump(parameters, open(config_folder + '/script_parameters.pkl', 'wb'))
		
		# Training
		run_wrapper(parameters, model_parameters, config_folder, DATABASE_PATH)
		
		# plot_weights(config_folder, logging)
		# generate_midi(config_folder, DATA_DIR, 20, None, logging)
		for track_path in track_paths:
			generate_midi_full_track_reference(config_folder, DATA_DIR, track_path, 5, logging)
else:
	# Already tested configs
	list_config_folders = glob.glob(result_folder + '/*')
	number_hp_config = max(0, N_HP_CONFIG - len(list_config_folders))
	for hp_config in range(number_hp_config):
		# Give a random ID and create folder
		ID_SET = False
		while not ID_SET:
			ID_config = str(random.randint(0, 2**25))
			config_folder = script_param['result_folder'] + '/' + ID_config
			if config_folder not in list_config_folders:
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
			config_folder = config_folder
			params = pkl.load(open(config_folder + '/config.pkl', "rb"))
			train.run_wrapper(params, config_folder)
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
	THEANO_FLAGS='device=gpu' python train.py '""" + config_folder + "'"

			with open(file_pbs, 'wb') as f:
				f.write(text_pbs)

			# Launch script
			subprocess.call('qsub ' + file_pbs, shell=True)

		# Update folder list
		list_config_folders.append(config_folder)