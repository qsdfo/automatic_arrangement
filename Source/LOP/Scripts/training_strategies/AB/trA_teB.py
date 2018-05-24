#!/usr/bin/env pyth
# -*- coding: utf8 -*-

import time
import random
import shutil
import os
import pickle as pkl

from LOP.Scripts.submit_job import submit_job
from LOP.Database.load_data import build_one_fold


class TS_trA_teB(object):
	
	def __init__(self, num_k_folds=10, config_folder=None, database_path=None, logger=None):
		"""Train, validate and test only on A
		"""
		self.num_k_folds = num_k_folds
		self.config_folder = config_folder
		self.database_path = database_path
		self.logger = logger
		# Important for reproducibility
		self.random_seed = 1234
		return

	def __build_folds(self, total_number_folds, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks, dataset_id):
		# Load files lists
		train_and_valid = pkl.load(open(self.database_path + '/train_and_valid_' + dataset_id + '.pkl', 'rb'))
		train_only = pkl.load(open(self.database_path + '/train_only_' + dataset_id + '.pkl', 'rb'))
		
		list_valid = list(train_and_valid.keys())
		list_train_only = list(train_only.keys())

		# Folds are built on files, not directly the indices
		# By doing so, we prevent the same file being spread over train, test and validate sets
		random.seed(self.random_seed)
		random.shuffle(list_files_valid)
		random.shuffle(list_files_train_only)

		if total_number_folds == -1:
			total_number_folds = len(list_files_valid)

		folds = []
		valid_names = []
		test_names = []

		# Build the list of split_matrices
		for current_fold_number in range(total_number_folds):
			one_fold, this_valid_names, this_test_names = build_one_fold(current_fold_number, total_number_folds, list_valid, list_train_only, 
				train_and_valid, train_only, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks)
			
			folds.append(one_fold)
			valid_names.append(this_valid_names)
			test_names.append(this_test_names)

		return folds, valid_names, test_names

	def merge_folds(K_folds_A, K_folds_B):
		K_folds = []
		for fold_A, fold_B in zip(K_folds_A, K_folds_B):
			fold = {'train': fold_A['train'] + fold_A['test'], # Use test as training since test is now another set
				'valid': fold_A['valid'],
				'test': fold_B['test']}
			K_folds.append(fold)
		return K_folds

	def get_folds(self, parameters, model_params):
		# Sadly, with the split of the database, shuffling the files can only be performed inside a same split
		self.logger.info('##### Building folds')
		# Load data and build K_folds
		time_load_0 = time.time()
		# K_folds[fold_index]['train','test' or 'valid'][index split]['batches' : [[234,14,54..],[..],[..]], 'matrices_path':[path_0,path_1,..]]
		import pdb; pdb.set_trace()
		if self.num_k_folds == 0:
			# this_K_folds, this_valid_names, this_test_names = build_folds(tracks_start_end, piano, orch, 10, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], RANDOM_SEED_FOLDS, logger_load=None)
			K_folds_A, valid_names_A, test_names_A = self.__build_folds(10, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], parameters["num_max_contiguous_blocks"], "A")
			K_folds_B, valid_names_B, test_names_B = self.__build_folds(10, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], parameters["num_max_contiguous_blocks"], "B")
			K_folds = merge_folds(K_folds_A, K_folds_B)
			self.K_folds = [K_folds[0]]
			self.valid_names = [valid_names_A[0]]
			self.test_names = [test_names_B[0]]
		elif self.num_k_folds == -1:
			raise Exception("num_k_folds = -1 Doesn't really make sense here")
		else:
			K_folds_A, valid_names_A, test_names_A = self.__build_folds(self.num_k_folds, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], parameters["num_max_contiguous_blocks"], "A")
			K_folds_B, valid_names_B, test_names_B = self.__build_folds(self.num_k_folds, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], parameters["num_max_contiguous_blocks"], "B")
			K_folds = merge_folds(K_folds_A, K_folds_B)
			self.self.K_folds = K_folds
			self.valid_names = valid_names_A
			self.test_names = test_names_B
		time_load = time.time() - time_load_0
		self.logger.info('TTT : Building folds took {} seconds'.format(time_load))
		return

	def submit_jobs(self, parameters, model_params, dimensions, track_paths_generation, save_bool, generate_bool, local):
		for K_fold_ind, K_fold in enumerate(self.K_folds):
			# Create fold folder
			config_folder_fold = self.config_folder + "/" + str(K_fold_ind)
			if os.path.isdir(config_folder_fold):
				shutil.rmtree(config_folder_fold)
			os.mkdir(config_folder_fold)
			# Submit worker
			submit_job(config_folder_fold, parameters, model_params, dimensions, K_fold, self.test_names[K_fold_ind], self.valid_names[K_fold_ind], track_paths_generation, save_bool, generate_bool, local, self.logger)
		return