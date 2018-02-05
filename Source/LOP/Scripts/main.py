#!/usr/bin/env pyth
# -*- coding: utf8 -*-

import cPickle as pkl
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

from train import train
from generate_midi import generate_midi
import config
from LOP.Database.load_data_k_folds import build_folds
from LOP.Utils.data_statistics import get_activation_ratio, get_mean_number_units_on
from load_matrices import load_matrices

# MODEL
# from LOP.Models.Real_time.Baseline.repeat import Repeat as Model
# from LOP.Models.Real_time.Baseline.mlp import MLP as Model
# from LOP.Models.Real_time.Baseline.mlp_K import MLP_K as Model
from LOP.Models.Real_time.LSTM_plugged_base import LSTM_plugged_base as Model
# from LOP.Models.Future_past_piano.Conv_recurrent.conv_recurrent_embedding_0 import Conv_recurrent_embedding_0 as Model

# NORMALIZER
from LOP.Utils.Normalization.no_normalization import no_normalization as Normalizer

GENERATE=False
SAVE=False
DEFINED_CONFIG=True  # HYPERPARAM ?
# For reproducibility
RANDOM_SEED_FOLDS=1234 # This is useful to use always the same fold split
RANDOM_SEED=None

def main():
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
	# Each config is a folder with a random ID (integer)
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
					# Clean
					# Only delete the files in the top folder and folds, but not the pretrained model
					for thing in os.listdir(config_folder):
						path_thing = os.path.join(config_folder, thing)
						if os.path.isfile(path_thing):
							os.remove(path_thing)
						elif thing != "pretraining":
							shutil.rmtree(path_thing)
					if not os.path.isdir(config_folder):    
						os.mkdir(config_folder) 
				else:
					raise Exception("Config not overwritten")
			config_loop(config_folder, model_parameters, parameters, DATABASE_PATH, track_paths_generation)
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

			config_loop(config_folder, model_parameters, parameters, DATABASE_PATH, track_paths_generation)

			# Update folder list
			list_config_folders.append(config_folder)


def config_loop(config_folder, model_params, parameters, database_path, track_paths_generation):
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
		# 1/
		# Pre-training then training on small db
		config_folder_pretraining = os.path.join(config_folder, 'pretraining')
		existing_pretrained_model = os.path.isdir(config_folder_pretraining)
		answer = ''
		if existing_pretrained_model:
			answer = raw_input("An existing pretrained model has been found. Press y if you want to pretrain it again : ")
			if answer=='y':
				shutil.rmtree(config_folder_pretraining)
		if (answer=='y') or (not existing_pretrained_model):
			os.makedirs(config_folder_pretraining)
			parameters['pretrained_model'] = None
			K_fold_ind = 0   # Only pretrain on the first K_fold
			train_wrapper(parameters, model_params, dimensions, config_folder_pretraining, 
						 (K_fold_ind, K_folds_pretraining[K_fold_ind]),
						 valid_names_pretraining[K_fold_ind], test_names_pretraining[K_fold_ind], track_paths_generation=[],
						 save_model=True, logger=logger_config)    

		for K_fold_ind, K_fold in enumerate(K_folds):
			parameters['pretrained_model'] = os.path.join(config_folder, 'pretraining', '0', 'model_acc', 'model')
			train_wrapper(parameters, model_params, dimensions, config_folder, 
						 (K_fold_ind, K_fold),
						 valid_names[K_fold_ind], test_names[K_fold_ind], track_paths_generation, 
						 save_model=SAVE, logger=logger_config)
		####################

	elif parameters['training_mode'] == 1:
		####################    
		# 2/ Append pretraining matrix (both train, valid and test parts) to the training data
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
			train_wrapper(parameters, model_params, dimensions, config_folder,
						  (K_fold_ind, new_K_fold),
						  valid_names[K_fold_ind], test_names[K_fold_ind], track_paths_generation, 
						  save_model=SAVE, logger=logger_config)
		####################
		
	elif parameters['training_mode'] == 2:
		####################
		# 3/
		# Full training only on the pretraining db
		# Used as a comparison, because the pretraining database is supposed to be "cleaner" than the "real one"
		parameters['pretrained_model'] = None
		for K_fold_ind, K_fold in enumerate(K_folds_pretraining):
			train_wrapper(parameters, model_params, dimensions, config_folder,
						(K_fold_ind, K_fold),
						valid_names_pretraining[K_fold_ind], test_names_pretraining[K_fold_ind], track_paths_generation,
						save_model=SAVE, logger=logger_config)
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
		# COuld be the weight for the weighted binary cross-entropy ???
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


def train_wrapper(parameters, model_params, dimensions, config_folder, 
				  K_fold_pair,
				  test_names, valid_names, track_paths_generation, 
				  save_model, logger):
	################################################################################
	################################################################################
	# TEST
	# percentage_training_set = model_params['percentage_training_set']
	# last_index = int(math.floor((percentage_training_set / float(100)) * len(train_folds)))
	# train_folds = train_folds[:last_index]
	################################################################################
	################################################################################
	K_fold_ind, K_fold = K_fold_pair
	train_folds = K_fold['train']
	valid_folds = K_fold['valid']
	valid_long_range_folds = K_fold['valid_long_range']

	config_folder_fold = config_folder + '/' + str(K_fold_ind)
	os.makedirs(config_folder_fold)
	# Write filenames of this split
	with open(os.path.join(config_folder_fold, "test_names.txt"), "wb") as f:
		for filename in test_names:
			f.write(filename + "\n")
	with open(os.path.join(config_folder_fold, "valid_names.txt"), "wb") as f:
		for filename in valid_names:
			f.write(filename + "\n")
	
	# Instanciate a normalizer for the input
	# normalizer = Normalizer(train_folds, n_components=20, whiten=True, parameters=parameters)
	# normalizer = Normalizer(train_folds, parameters)
	normalizer = Normalizer(dimensions)
	pkl.dump(normalizer, open(os.path.join(config_folder_fold, 'normalizer.pkl'), 'wb'))
	dimensions['piano_transformed_dim'] = normalizer.transformed_dim

	# Compute training data's statistics for improving learning (e.g. weighted Xent)
	activation_ratio = get_activation_ratio(train_folds, dimensions['orch_dim'], parameters)
	mean_number_units_on = get_mean_number_units_on(train_folds, parameters)
	# It's okay to add this value to the parameters now because we don't need it for persistency, 
	# this is only training regularization
	model_params['activation_ratio'] = activation_ratio
	parameters['activation_ratio'] = activation_ratio
	model_params['mean_number_units_on'] = mean_number_units_on
	
	########################################################
	# Persistency
	pkl.dump(model_params, open(config_folder + '/model_params.pkl', 'wb'))
	pkl.dump(Model.is_keras(), open(config_folder + '/is_keras.pkl', 'wb'))
	pkl.dump(parameters, open(config_folder + '/script_parameters.pkl', 'wb'))

	############################################################
	# Update train_param and model_param dicts with new information from load data
	############################################################
	def count_number_batch(fold):
		counter = 0
		for batches in fold.values():
			counter += len(batches)
		return counter
	n_train_batches = count_number_batch(train_folds)
	n_val_batches = count_number_batch(valid_folds)
	n_val_long_range_batches = count_number_batch(valid_long_range_folds)

	logger.info((u'# n_train_batch :  {}'.format(n_train_batches)).encode('utf8'))
	logger.info((u'# n_val_batch :  {}'.format(n_val_batches)).encode('utf8'))
	logger.info((u'# n_val_long_range_batch :  {}'.format(n_val_long_range_batches)).encode('utf8'))

	parameters['n_train_batches'] = n_train_batches
	parameters['n_val_batches'] = n_val_batches
	parameters['n_val_long_range_batches'] = n_val_long_range_batches

	############################################################
	# Instanciate model and save folder
	############################################################
	model = Model(model_params, dimensions)
	for measure_name in parameters["save_measures"]:
		os.mkdir(config_folder_fold + '/model_' + measure_name)
	
	############################################################
	# Train
	############################################################
	time_train_0 = time.time()
	valid_tabs, best_epoch, valid_tabs_LR, best_epoch_LR = train(model, train_folds, valid_folds, valid_long_range_folds, normalizer, parameters, config_folder_fold, time_train_0, logger)
	time_train_1 = time.time()
	training_time = time_train_1-time_train_0
	logger.info('TTT : Training data took {} seconds'.format(training_time))
	logger.info((u'# Best loss obtained at epoch :  {}'.format(best_epoch['loss'])).encode('utf8'))
	logger.info((u'# Loss :  {}'.format(valid_tabs['loss'][best_epoch['loss']])).encode('utf8'))
	logger.info((u'# Accuracy :  {}'.format(valid_tabs['accuracy'][best_epoch['accuracy']])).encode('utf8'))
	logger.info((u'###################\n').encode('utf8'))

	############################################################
	# Write result in a txt file
	############################################################
	os.mkdir(os.path.join(config_folder_fold, 'results_short_range'))
	os.mkdir(os.path.join(config_folder_fold, 'results_long_range'))
	
	# Short range
	for measure_name, measure_curve in valid_tabs.iteritems():
		np.savetxt(os.path.join(config_folder_fold, 'results_short_range', measure_name + '.txt'), measure_curve, fmt='%.6f')
		with open(os.path.join(config_folder_fold, 'results_short_range', measure_name + '_best_epoch.txt'), 'wb') as f:
			f.write("{:d}".format(best_epoch[measure_name]))
	# Long range
	for measure_name, measure_curve in valid_tabs_LR.iteritems():
		np.savetxt(os.path.join(config_folder_fold, 'results_long_range', measure_name + '.txt'), measure_curve, fmt='%.6f')
		with open(os.path.join(config_folder_fold, 'results_long_range', measure_name + '_best_epoch.txt'), 'wb') as f:
			f.write("{:d}".format(best_epoch[measure_name]))

	# Generating
	if GENERATE:
		generate_wrapper(config_folder_fold, track_paths_generation, logger)
	if not save_model:
		for measure_name in parameters['save_measures']:
			shutil.rmtree(config_folder_fold + '/model_' + measure_name)
		########################################################
	return

def generate_wrapper(config_folder, track_paths_generation, logger):
	for score_source in track_paths_generation:
			generate_midi(config_folder, score_source, number_of_version=3, duration_gen=100, rhythmic_reconstruction=False, logger_generate=logger)
	return

if __name__ == '__main__':
	main()
