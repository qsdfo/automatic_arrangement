#!/usr/bin/env pyth
# -*- coding: utf8 -*-

import pickle as pkl
import subprocess
import random
import logging
import glob
import re
import os
import shutil
import time
import numpy as np
import hyperopt
import copy

from import_functions import import_model, import_training_strategy
import config

GENERATE=True
SAVE=False
DEFINED_CONFIG=True  # HYPERPARAM ?
# For reproducibility
RANDOM_SEED_FOLDS=1234 # This is useful to use always the same fold split
RANDOM_SEED=None

def main():
	model_name = config.model()
	Model = import_model.import_model(model_name)

	# DATABASE
	DATABASE = config.data_name()
	DATABASE_PATH = config.data_root() + "/" + DATABASE
	# RESULTS
	result_folder =  config.result_root() + '/' + DATABASE + '/' + Model.name()
	if not os.path.isdir(result_folder):
		os.makedirs(result_folder)

	# Parameters
	parameters = config.parameters(result_folder)
	if os.path.isfile(DATABASE_PATH + '/binary_piano'):
		parameters["binarize_piano"] = True
	else:
		parameters["binarize_piano"] = False
	if os.path.isfile(DATABASE_PATH + '/binary_orch'):
		parameters["binarize_orch"] = True 
	else:
		parameters["binarize_orch"] = False

	parameters["model_name"] = model_name

	# Load the database metadata and add them to the script parameters to keep a record of the data processing pipeline
	parameters.update(pkl.load(open(DATABASE_PATH + '/metadata.pkl', 'rb')))

	############################################################
	# Logging
	############################################################
	# log file
	log_file_path = 'log'
	# set up logging to file - see previous section for more details
	logging.basicConfig(level=logging.INFO,
						format='%(asctime)s %(levelname)-8s %(message)s',
						datefmt='%m-%d %H:%M',
						filename=log_file_path,
						filemode='w')
	# define a Handler which writes INFO messages or higher to the sys.stderr
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	# set a format which is simpler for console use
	formatter = logging.Formatter('%(levelname)-8s %(message)s')
	# tell the handler to use this format
	console.setFormatter(formatter)
	# add the handler to the root logger
	logging.getLogger('').addHandler(console)

	# Now, we can log to the root logger, or any other logger. First the root...
	logging.info('#'*60)
	logging.info('#'*60)
	logging.info('#'*60)
	logging.info('* L * O * P *')
	logging.info('** Model : ' + Model.name())
	for k, v in parameters.items():
		logging.info('** ' + k + ' : ' + str(v))
	logging.info('#'*60)
	logging.info('#'*60)

	############################################################
	# Hyper parameter space
	############################################################
	# Two cases :
	# 1/ Random search
	model_parameters_space = Model.get_hp_space()
	# 2/ Defined configurations
	configs = config.import_configs()
	
	# On from each database and each set
	track_paths_generation = [
		# Bouliane train
		config.database_root() + '/bouliane/0',
		# Bouliane test
		config.database_root() + '/bouliane/17',
		# Bouliane valid
		config.database_root() + '/bouliane/16',
		# Spotify train
		config.database_root() + '/hand_picked_Spotify/0',
		# Spotify test
		config.database_root() + '/hand_picked_Spotify/21',
		# Spotify valid
		config.database_root() + '/hand_picked_Spotify/20',
		# Liszt train
		config.database_root() + '/liszt_classical_archives/0',
		# Liszt test
		config.database_root() + '/liszt_classical_archives/17',
		# Liszt valid
		config.database_root() + '/liszt_classical_archives/16'
	]

	############################################################
	# Grid search loop
	############################################################
	# Organisation :
	# Each config is a folder with a random ID (integer)
	# In eahc of this folder there is :
	#    - a config.pkl file with the hyper-parameter space
	#    - a result.txt file with the result
	# The result.csv file containing id;result is created from the directory, rebuilt from time to time

	if DEFINED_CONFIG:
		for config_id, model_parameters in configs.items():
			config_folder = parameters['result_folder'] + '/' + config_id
			if os.path.isdir(config_folder):
				shutil.rmtree(config_folder)
			os.mkdir(config_folder)
			config_loop(Model, config_folder, model_parameters, parameters, DATABASE_PATH, track_paths_generation)
	else:
		# Already tested configs
		list_config_folders = glob.glob(result_folder + '/*')
		number_hp_config = max(0, parameters["max_hyperparam_configs"] - len(list_config_folders))
		for hp_config in range(number_hp_config):
			# Give a random ID and create folder
			ID_SET = False
			while not ID_SET:
				ID_config = str(random.randint(0, 2**25))
				config_folder = parameters['result_folder'] + '/' + ID_config
				if config_folder not in list_config_folders:
					ID_SET = True
			os.mkdir(config_folder)

			# Sample model parameters from hyperparam space
			model_parameters = hyperopt.pyll.stochastic.sample(model_parameters_space)

			config_loop(Model, config_folder, model_parameters, parameters, DATABASE_PATH, track_paths_generation)

			# Update folder list
			list_config_folders.append(config_folder)
	return


def config_loop(Model, config_folder, model_params, parameters, database_path, track_paths_generation):
	# New logger
	log_file_path = config_folder + '/' + 'log.txt'
	with open(log_file_path, 'wb') as f:
		f.close()
	formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
	logger_config = logging.getLogger(config_folder)
	hdlr = logging.FileHandler(log_file_path)
	hdlr.setFormatter(formatter)
	logger_config.addHandler(hdlr)
	
	# Prompt model parameters
	logger_config.info('#'*60)
	logger_config.info('#### ' + config_folder)
	logger_config.info('#### Model parameters')
	logger_config.info('** Model : ' + Model.name())
	for k, v in model_params.items():
		logger_config.info('** ' + k + ' : ' + str(v))

	# Get dimensions of batches (will be the same for pretraining)
	dimensions = {'temporal_order': model_params['temporal_order'],
		'orch_dim': parameters['N_orchestra']}
	if parameters["embedded_piano"]:
		dimensions['piano_embedded_dim'] = parameters["N_piano_embedded"]
	else:
		dimensions['piano_embedded_dim'] = parameters["N_piano"]
	
	if parameters["duration_piano"]:
		dimensions['piano_input_dim'] = dimensions['piano_embedded_dim'] + 1
	else:
		dimensions['piano_input_dim'] = dimensions['piano_embedded_dim']
	pkl.dump(dimensions, open(config_folder + '/dimensions.pkl', 'wb'))

	Training_strategy = import_training_strategy.import_training_strategy(parameters["training_strategy"])
	training_strategy = Training_strategy(num_k_folds=10, config_folder=config_folder, database_path=database_path, logger=logger_config)
	with open(config_folder + "/K_fold_strategy", 'w') as ff:
		ff.write(training_strategy.name() + '\n')
	training_strategy.get_folds(parameters, model_params)
	training_strategy.submit_jobs(parameters, model_params, dimensions, track_paths_generation, SAVE, GENERATE, config.local())
	return

if __name__ == '__main__':
	main()
