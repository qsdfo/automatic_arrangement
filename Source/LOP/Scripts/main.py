#!/usr/bin/env python
# -*- coding: utf8 -*-

import cPickle as pkl
import re
import math
import random
import logging
import glob
import os
import shutil
import time
import numpy as np
import hyperopt

from train import train
from generate_midi import generate_midi
import config
from LOP.Database.load_data import load_data_train, load_data_valid, load_data_test

# MODEL
from LOP.Models.Real_time.Baseline.mlp_K import MLP_K as Model

GENERATE = False

def main():
	# DATABASE
	DATABASE = "Data__event_level8__0"
	DATABASE_PATH = config.data_root() + "/" + DATABASE
	# HYPERPARAM ?
	DEFINED_CONFIG = False
	# RESULTS
	result_folder =  config.result_root() + '/' + DATABASE + '/' + Model.name()
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
		"max_iter": 100,            # nb max of iterations when training 1 configuration of hparams (~200)
		"walltime": 11,             # in hours
		# Validation
		"min_number_iteration": 10,
		"validation_order": 2,
		"number_strips": 3,
		# Hyperopt
		"max_hyperparam_configs": 20,            # number of hyper-parameter configurations evaluated
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
	logging.info('#'*60)
	logging.info('#'*60)
	logging.info('#'*60)
	logging.info('* L * O * P *')
	logging.info((u'** Model : ' + Model.name()).encode('utf8'))
	logging.info((u'** Optimization technic : ##########').encode('utf8'))
	logging.info((u'** Temporal granularity : ' + parameters['temporal_granularity']).encode('utf8'))
	logging.info((u'** Quantization : ' + str(parameters['quantization'])).encode('utf8'))
	logging.info((u'** Binary piano : ' + str(parameters["binarize_piano"])).encode('utf8'))
	logging.info((u'** Binary orchestra : ' + str(parameters["binarize_orchestra"])).encode('utf8'))
	logging.info((u'** Result folder : ' + result_folder).encode('utf8'))
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
	
	# configs = {}
	# for k in range(6,10):
	# 	# Run 10 times each config, to have a statistically relevent plot
	# 	for j in range(10):
	# 		temp = {}
	# 		temp = dict(configs__["0"])
	# 		temp['percentage_training_set'] = (k+1)*10
	# 		config_ID = (k+1) * 10 + j
	# 		configs[str(config_ID)] = temp
	
	# On from each database and each set
	track_paths_generation = [
		# Bouliane train
		config.database_root() + '/LOP_database_06_09_17/bouliane/0',
		# Bouliane test
		config.database_root() + '/LOP_database_06_09_17/bouliane/17',
		# Bouliane valid
		config.database_root() + '/LOP_database_06_09_17/bouliane/16',
		# Spotify train
		config.database_root() + '/LOP_database_06_09_17/hand_picked_Spotify/0',
		# Spotify test
		config.database_root() + '/LOP_database_06_09_17/hand_picked_Spotify/21',
		# Spotify valid
		config.database_root() + '/LOP_database_06_09_17/hand_picked_Spotify/20',
		# Liszt train
		config.database_root() + '/LOP_database_06_09_17/liszt_classical_archives/0',
		# Liszt test
		config.database_root() + '/LOP_database_06_09_17/liszt_classical_archives/17',
		# Liszt valid
		config.database_root() + '/LOP_database_06_09_17/liszt_classical_archives/16'
	]

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
				# continue
				user_input = raw_input(config_folder + " folder already exists. Type y to overwrite : ")
				if user_input == 'y':
					shutil.rmtree(config_folder)
					os.mkdir(config_folder)	
				else:
					raise Exception("Config not overwritten")
			config_loop(config_folder, model_parameters, parameters, DATABASE_PATH, track_paths_generation)
	else:
		# Already tested configs
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

			# Find a point in space that has never been tested
			# UNTESTED_POINT_FOUND = False
			# while not UNTESTED_POINT_FOUND:
			# 	model_parameters = hyperopt.pyll.stochastic.sample(model_parameters_space)
			# 	# Check that this point in space has never been tested
			# 	# By looking in all directories and reading the config.pkl file

			# 	UNTESTED_POINT_FOUND = True
			# 	for dirname in list_config_folders:
			# 		this_config = pkl.load(open(dirname + '/model_parameters.pkl', 'rb'))
			# 		if space == this_config:
			# 			UNTESTED_POINT_FOUND = False
			# 			break

			# Sample model parameters from hyperparam space
			model_parameters = hyperopt.pyll.stochastic.sample(model_parameters_space)

			config_loop(config_folder, model_parameters, parameters, DATABASE_PATH, track_paths_generation)

			# Update folder list
			list_config_folders.append(config_folder)


def config_loop(config_folder, model_parameters, parameters, database_path, track_paths_generation):
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
	logger_config.info((u'** Model : ' + Model.name()).encode('utf8'))
	for k, v in model_parameters.iteritems():
		logger_config.info((u'** ' + k + ' : ' + str(v)).encode('utf8'))
	# Persistency
	pkl.dump(model_parameters, open(config_folder + '/model_parameters.pkl', 'wb'))
	pkl.dump(Model.is_keras(), open(config_folder + '/is_keras.pkl', 'wb'))
	pkl.dump(parameters, open(config_folder + '/script_parameters.pkl', 'wb'))
	# Training
	train_wrapper(parameters, model_parameters, config_folder, database_path, logger_config)
	# Generating
	if GENERATE:
		generate_wrapper(config_folder, track_paths_generation, logger_config)
	logger_config.info("#"*60)
	logger_config.info("#"*60)
	return


def train_wrapper(parameters, model_params, config_folder, data_folder, logger):
	############################################################
	# Load data
	############################################################
	time_load_0 = time.time()
	piano_train, orch_train, train_index \
		= load_data_train(data_folder,
						  model_params['temporal_order'],
						  model_params['batch_size'],
						  skip_sample=parameters['skip_sample'],
						  avoid_silence=parameters['avoid_silence'],
						  binarize_piano=parameters["binarize_piano"],
						  binarize_orchestra=parameters["binarize_orchestra"],
						  logger_load=logger)
	piano_valid, orch_valid, valid_index \
		= load_data_valid(data_folder,
						  model_params['temporal_order'],
						  model_params['batch_size'],
						  skip_sample=parameters['skip_sample'],
						  avoid_silence=True,
						  binarize_piano=parameters["binarize_piano"],
						  binarize_orchestra=parameters["binarize_orchestra"],
						  logger_load=logger)
	piano_test, orch_test, _, _ \
		= load_data_test(data_folder,
						 model_params['temporal_order'],
						 model_params['batch_size'],
						 skip_sample=parameters['skip_sample'],
						 avoid_silence=True,
						 binarize_piano=parameters["binarize_piano"],
						 binarize_orchestra=parameters["binarize_orchestra"],
						 logger_load=logger)
	time_load_1 = time.time()

	################################################################################
	################################################################################
	################################################################################
	################################################################################
	# TEST
	# percentage_training_set = model_params['percentage_training_set']
	# last_index = int(math.floor((percentage_training_set / float(100)) * len(train_index)))
	# train_index = train_index[:last_index]
	################################################################################
	################################################################################
	################################################################################
	################################################################################################################################################################

	############################################################
	# Get dimensions of batches
	############################################################
	piano_dim = piano_train.shape[1]
	orch_dim = orch_train.shape[1]
	dimensions = {'batch_size': model_params['batch_size'],
				  'temporal_order': model_params['temporal_order'],
				  'piano_dim': piano_dim,
				  'orch_dim': orch_dim}
	pkl.dump(dimensions, open(config_folder + '/dimensions.pkl', 'wb'))

	############################################################
	# Update train_param and model_param dicts with new information from load data
	############################################################
	n_train_batches = len(train_index)
	n_val_batches = len(valid_index)

	logger.info((u'##### Data').encode('utf8'))
	logger.info((u'# n_train_batch :  {}'.format(n_train_batches)).encode('utf8'))
	logger.info((u'# n_val_batch :  {}'.format(n_val_batches)).encode('utf8'))
	logger.info('TTT : Loading data took {} seconds'.format(time_load_1-time_load_0))

	parameters['n_train_batches'] = n_train_batches
	parameters['n_val_batches'] = n_val_batches

	# Class normalization
	notes_activation = orch_train.sum(axis=0)
	notes_activation_norm = notes_activation.mean() / (notes_activation+1e-10)
	class_normalization = np.maximum(1, np.minimum(20, notes_activation_norm))
	model_params['class_normalization'] = class_normalization

	# Other kind of regularization
	L_train = orch_train.shape[0]
	mean_notes_activation = notes_activation / L_train
	mean_notes_activation = np.where(mean_notes_activation == 0, 1. / L_train, mean_notes_activation)
	model_params['mean_notes_activation'] = mean_notes_activation

	############################################################
	# Instanciate model and save folder
	############################################################
	model = Model(model_params, dimensions)
	os.mkdir(config_folder + '/model/')

	############################################################
	# Train
	############################################################
	time_train_0 = time.time()
	loss_val, accuracy_val, best_epoch = train(model,
									piano_train, orch_train, train_index,
									piano_valid, orch_valid, valid_index,
									parameters, config_folder, time_train_0, logger)

	time_train_1 = time.time()
	training_time = time_train_1-time_train_0
	logger.info('TTT : Training data took {} seconds'.format(training_time))
	logger.info((u'# Best model obtained at epoch :  {}'.format(best_epoch)).encode('utf8'))
	logger.info((u'# Loss :  {}'.format(loss_val)).encode('utf8'))
	logger.info((u'# Accuracy :  {}'.format(accuracy_val)).encode('utf8'))
	logger.info((u'###################\n').encode('utf8'))

	############################################################
	# Write result in a txt file
	############################################################
	result_file_path = config_folder + '/result.csv'
	with open(result_file_path, 'wb') as f:
		f.write("accuracy;loss\n" + str(accuracy_val) + ";" + str(loss_val))

	return

def generate_wrapper(config_folder, track_paths_generation, logger):
	for score_source in track_paths_generation:
			generate_midi(config_folder, score_source, 3, logger)
	

if __name__ == '__main__':
	main()