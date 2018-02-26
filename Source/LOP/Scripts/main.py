#!/usr/bin/env pyth
# -*- coding: utf8 -*-

import cPickle as pkl
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

from import_functions import import_model
import train_wrapper
import config
from LOP.Database.load_data_k_folds import build_folds
from load_matrices import load_matrices

MODEL_NAME="LSTM_plugged_base"
GENERATE=True
SAVE=False
DEFINED_CONFIG=True  # HYPERPARAM ?
# For reproducibility
RANDOM_SEED_FOLDS=1234 # This is useful to use always the same fold split
RANDOM_SEED=None

def main():
	Model = import_model.import_model(MODEL_NAME)

	# DATABASE
	DATABASE = config.data_name()
	DATABASE_PATH = config.data_root() + "/" + DATABASE
	# RESULTS
	result_folder =  config.result_root() + '/' + DATABASE + '/' + Model.name()
	if not os.path.isdir(result_folder):
		os.makedirs(result_folder)
	# Parameters
	parameters = config.parameters(result_folder)

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
	logging.info((u'** Model : ' + Model.name()).encode('utf8'))
	for k, v in parameters.iteritems():
		logging.info((u'** ' + k + ' : ' + str(v)).encode('utf8'))
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
		for config_id, model_parameters in configs.iteritems():
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
	logger_config.info((u'** Model : ' + Model.name()).encode('utf8'))
	for k, v in model_params.iteritems():
		logger_config.info((u'** ' + k + ' : ' + str(v)).encode('utf8'))
	
	# Get tvt splits. Dictionnary form :
	# K_folds[fold_index]['train','test' or 'valid'][split_matrix_index]
	K_folds, valid_names, test_names, dimensions = \
		get_folds(database_path, parameters['k_folds'], parameters, model_params, suffix="", logger=logger_config)
	pretraining_bool = re.search ("_pretraining", config.data_name())
	if pretraining_bool:
		K_folds_pretraining, valid_names_pretraining, test_names_pretraining, _ = \
			get_folds(database_path, parameters['k_folds'], parameters, model_params, suffix="_pretraining", logger=logger_config)
	pkl.dump(dimensions, open(config_folder + '/dimensions.pkl', 'wb'))
	
	# Three options : pre-training then training or just concatenate everything
	if parameters['training_mode'] == 0:
		####################
		# 0/
		# Pre-training then training on small db
		config_folder_pretraining = os.path.join(config_folder, 'pretraining')
		if os.path.isdir(config_folder_pretraining):
			shutil.rmtree(config_folder_pretraining)
		os.makedirs(config_folder_pretraining)
		parameters['pretrained_model'] = None

		# This is gonna be a problem for Guillimin. Main will have to wait for the end of the pretraining worker
		# Write pbs script
		submit_job(config_folder_pretraining, parameters, model_params, dimensions, K_folds_pretraining[0], test_names_pretraining[0], valid_names_pretraining[0],
			track_paths_generation, True, logger_config)

		for K_fold_ind, K_fold in enumerate(K_folds):
			parameters['pretrained_model'] = os.path.join(config_folder, 'pretraining', '0', 'model_accuracy')
			# Create fold folder
			config_folder_fold = config_folder + "/" + str(K_fold_ind)
			if os.path.isdir(config_folder_fold):
				shutil.rmtree(config_folder_fold)
			os.mkdir(config_folder_fold)
			# Submit workers
			submit_job(config_folder_fold, parameters, model_params, dimensions, K_fold, test_names[K_fold_ind], valid_names[K_fold_ind],
				track_paths_generation, SAVE, logger_config)
			# train_wrapper(parameters, model_params, dimensions, config_folder, 
			# 			 (K_fold_ind, K_fold),
			# 			 valid_names[K_fold_ind], test_names[K_fold_ind], track_paths_generation, 
			# 			 save_model=SAVE, logger=logger_config)
		####################

	elif parameters['training_mode'] == 1:
		####################    
		# 1/ Append pretraining matrix (both train, valid and test parts) to the training data
		for K_fold_ind, K_fold in enumerate(K_folds):
			parameters['pretrained_model'] = None
			if pretraining_bool:
				new_K_fold = copy.deepcopy(K_fold)
				paths_pretraining_matrices = K_folds_pretraining[0]['train'].keys()
				for paths_pretraining_matrix in paths_pretraining_matrices:
					indices_from_pretraining = K_folds_pretraining[0]['train'][paths_pretraining_matrix] + K_folds_pretraining[0]['test'][paths_pretraining_matrix] + K_folds_pretraining[0]['valid'][paths_pretraining_matrix]
					new_K_fold['train'][paths_pretraining_matrix] = indices_from_pretraining
					# Note that valid_names and test_names don't change
			else:
				new_K_fold = K_fold
			# Create fold folder
			config_folder_fold = config_folder + "/" + str(K_fold_ind)
			if os.path.isdir(config_folder_fold):
				shutil.rmtree(config_folder_fold)
			os.mkdir(config_folder_fold)
			# Submit worker
			submit_job(config_folder_fold, parameters, model_params, dimensions, K_fold, test_names[K_fold_ind], valid_names[K_fold_ind],
				track_paths_generation, SAVE, logger_config)
			# train_wrapper(parameters, model_params, dimensions, config_folder,
			# 			  (K_fold_ind, new_K_fold),
			# 			  valid_names[K_fold_ind], test_names[K_fold_ind], track_paths_generation, 
			# 			  save_model=SAVE, logger=logger_config)
		####################
		
	elif parameters['training_mode'] == 2:
		####################
		# 2/
		# Full training only on the pretraining db
		# Used as a comparison, because the pretraining database is supposed to be "cleaner" than the "real one"
		parameters['pretrained_model'] = None
		for K_fold_ind, K_fold in enumerate(K_folds_pretraining):
			# Create fold folder
			config_folder_fold = config_folder + "/" + str(K_fold_ind)
			if os.path.isdir(config_folder_fold):
				shutil.rmtree(config_folder_fold)
			os.mkdir(config_folder_fold)
			# Submit worker
			submit_job(config_folder_fold, parameters, model_params, dimensions, K_fold, test_names[K_fold_ind], valid_names[K_fold_ind],
				track_paths_generation, SAVE, logger_config)
			# train_wrapper(parameters, model_params, dimensions, config_folder,
			# 			(K_fold_ind, K_fold),
			# 			valid_names_pretraining[K_fold_ind], test_names_pretraining[K_fold_ind], track_paths_generation,
			# 			save_model=SAVE, logger=logger_config)
		####################
	else:
		raise Exception("Not a training mode")

	logger_config.info("#"*60)
	logger_config.info("#"*60)
	return


def get_folds(database_path, num_k_folds, parameters, model_params, suffix=None, logger=None):
	# Sadly, with the split of the database, shuffling the files can only be performed inside a same split
	logger.info((u'##### Building folds').encode('utf8'))
	
	# Load data and build K_folds
	time_load_0 = time.time()

	## Load the matrices
	piano_files = glob.glob(database_path + '/piano' + suffix + '_[0-9]*.npy')
	
	# K_folds[fold_index]['train','test' or 'valid'][matrix_path]
	K_fold = [
			{'train': [],
			'test': [],
			'valid': [],
			'valid_long_range': []}
		]
	K_folds = []
	valid_names = []
	test_names = []
	init_folds = True

	for counter_split, piano_file in enumerate(piano_files):

		piano, orch, duration_piano, mask_orch, tracks_start_end = load_matrices(piano_file, parameters)

		################################################################################
		################################################################################
		################################################################################
		# Could be the weight for the weighted binary cross-entropy ???
		# stat_for_Xent_weight = orch.shape[0]*orch.shape[1]/orch.sum()
		################################################################################
		################################################################################
		################################################################################

		if num_k_folds == 0:
			this_K_folds, this_valid_names, this_test_names = build_folds(tracks_start_end, piano, orch, 10, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], RANDOM_SEED_FOLDS, logger_load=None)
			this_K_folds = [K_folds[0]]
			this_valid_names = [this_valid_names[0]]
			this_test_names = [this_test_names[0]]
		elif num_k_folds == -1:
			this_K_folds, this_valid_names, this_test_names = build_folds(tracks_start_end, piano, orch, -1, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], RANDOM_SEED_FOLDS, logger_load=None)
		else:
			this_K_folds, this_valid_names, this_test_names = build_folds(tracks_start_end, piano, orch, num_k_folds, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], RANDOM_SEED_FOLDS, logger_load=None)
		
		for k_ind, fold in enumerate(this_K_folds):
			if init_folds:
				K_folds.append({'train' : {}, 'test': {}, 'valid': {}, 'valid_long_range': {}})
				valid_names.append([])
				test_names.append([])
			for split_name, batches in fold.iteritems():
				K_folds[k_ind][split_name][piano_file] = batches
			valid_names[k_ind].extend(this_valid_names[k_ind])
			test_names[k_ind].extend(this_test_names[k_ind])
		
		init_folds = False

	time_load = time.time() - time_load_0

	## Get dimensions of batches (will be the same for pretraining)
	piano_dim = piano.shape[1]
	orch_dim = orch.shape[1]
	dimensions = {'temporal_order': model_params['temporal_order'],
				  'piano_dim': piano_dim,
				  'orch_dim': orch_dim}
	logger.info('TTT : Building folds took {} seconds'.format(time_load))
	return K_folds, valid_names, test_names, dimensions


def submit_job(config_folder_fold, parameters, model_params, dimensions, K_fold, test_names, valid_names,
	track_paths_generation, save_model, logger):
	
	if config.local():
		train_wrapper.train_wrapper(parameters, model_params, MODEL_NAME, dimensions, config_folder_fold,
				  K_fold,
				  test_names, valid_names, track_paths_generation, 
				  save_model, GENERATE, logger)
	else:
		context_folder = config_folder_fold + '/context'
		os.mkdir(context_folder)

		# Save all the arguments of the wrapper script
		pkl.dump(parameters, open(context_folder + "/parameters.pkl", 'wb')) 
		pkl.dump(model_params, open(context_folder + '/model_params.pkl', 'wb'))
		pkl.dump(MODEL_NAME, open(context_folder + '/model_name.pkl', 'wb'))
		pkl.dump(dimensions , open(context_folder + '/dimensions.pkl', 'wb'))
		pkl.dump(K_fold, open(context_folder + '/K_fold.pkl', 'wb'))
		pkl.dump(test_names, open(context_folder + '/test_names.pkl', 'wb'))
		pkl.dump(valid_names , open(context_folder + '/valid_names.pkl', 'wb'))
		pkl.dump(track_paths_generation, open(context_folder + '/track_paths_generation.pkl', 'wb'))
		pkl.dump(save_model , open(context_folder + '/save_model.pkl', 'wb'))
		pkl.dump(GENERATE , open(context_folder + '/generate_bool.pkl', 'wb'))
		
		# Write pbs script
		file_pbs = context_folder + '/submit.pbs'

		text_pbs = """#!/bin/bash

#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l pmem=4000m
#PBS -l walltime=""" + str(parameters['walltime']) + """:00:00

module load foss/2015b
module load Tensorflow/1.0.0-Python-2.7.12
source ~/Virtualenvs/lop/bin/activate

SRC=/home/crestel/automatic_arrangement/Source/LOP/Scripts
cd $SRC
python train_wrapper.py '""" + config_folder_fold + "'"

		with open(file_pbs, 'wb') as f:
			f.write(text_pbs)

		# Launch script
		subprocess.call('qsub ' + file_pbs, shell=True)
		return

if __name__ == '__main__':
	main()
