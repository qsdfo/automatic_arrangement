#!/usr/bin/env python
# -*- coding: utf8 -*-

""" This is the script called by qsub on Guillimin
"""

import os
import sys
import time
import shutil
import numpy as np
import cPickle as pkl
import import_model
from train import train
from generate_midi import generate_midi
from LOP.Utils.data_statistics import get_activation_ratio, get_mean_number_units_on

# NORMALIZER
from LOP.Utils.Normalization.no_normalization import no_normalization as Normalizer

def train_wrapper(parameters, model_params, model_name,
	dimensions, config_folder_fold, K_fold,
	test_names, valid_names, track_paths_generation, 
	save_model, generate_bool, logger):
	
	Model = import_model.get_model(model_name)

	train_folds = K_fold['train']
	valid_folds = K_fold['valid']
	valid_long_range_folds = K_fold['valid_long_range']
	
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
	pkl.dump(model_params, open(config_folder_fold + '/model_params.pkl', 'wb'))
	pkl.dump(Model.is_keras(), open(config_folder_fold + '/is_keras.pkl', 'wb'))
	pkl.dump(parameters, open(config_folder_fold + '/script_parameters.pkl', 'wb'))

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
	if generate_bool:
		generate_wrapper(config_folder_fold, track_paths_generation, logger)
	if not save_model:
		for measure_name in parameters['save_measures']:
			shutil.rmtree(config_folder_fold + '/model_' + measure_name)
		########################################################
	return

def generate_wrapper(config_folder_fold, track_paths_generation, logger):
	for score_source in track_paths_generation:
			generate_midi(config_folder_fold, score_source, number_of_version=3, duration_gen=100, rhythmic_reconstruction=False, logger_generate=logger)
	return


if __name__ == '__main__':

	config_folder_fold = sys.argv[1]
	context_folder = config_folder_fold + "/context"
	
	# Get parameters
	parameters =  pickle.load(open(context_folder + "/parameters.pkl","rb"))
	model_params = pickle.load(open(context_folder + "/model_params.pkl","rb"))
	model_name = pickle.load(open(context_folder + "/model_name.pkl","rb"))
	dimensions = pickle.load(open(context_folder + "/dimensions.pkl","rb")) 
	K_fold = pickle.load(open(context_folder + "/K_fold.pkl","rb"))
	test_names = pickle.load(open(context_folder + "/test_names.pkl","rb"))
	valid_names = pickle.load(open(context_folder + "/valid_names.pkl","rb")) 
	track_paths_generation = pickle.load(open(context_folder + "/track_paths_generation.pkl","rb"))
	save_model = pickle.load(open(context_folder + "/save_model.pkl","rb"))
	generate_bool = pickle.load(open(context_folder + "/generate_bool.pkl","rb"))

	import logging
	formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
	logger_config = logging.getLogger(config_folder)
	hdlr = logging.FileHandler(log_file_path)
	hdlr.setFormatter(formatter)
	logger_config.addHandler(hdlr)

	train_wrapper(parameters, model_params, model_name, 
		dimensions, config_folder_fold, K_fold,
		test_names, valid_names, track_paths_generation, 
		save_model, generate_bool, logger_config)