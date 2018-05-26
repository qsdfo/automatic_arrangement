#!/usr/bin/env pyth
# -*- coding: utf8 -*-

import time
import random
import shutil
import os
import pickle as pkl

from LOP.Scripts.submit_job import submit_job
from LOP.Database.load_data import build_one_fold


class TS_trC__B__A_teA(object):
	
	def __init__(self, num_k_folds=10, config_folder=None, database_path=None, logger=None):
		"""Train, validate and test only on A
		"""
		self.num_k_folds = num_k_folds
		self.config_folder = config_folder
		self.database_path = database_path
		self.logger = logger
		# Important for reproducibility
		self.random_inst = random.Random()
		self.random_inst.seed(1234)
		return

	@staticmethod
	def name():
		return "TS_trC__B__A_teA"

	def __build_folds(self, total_number_folds, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks):
		# Load files lists
		t_dict_A = pkl.load(open(self.database_path + '/train_only_A.pkl', 'rb'))
		tvt_dict_A = pkl.load(open(self.database_path + '/train_and_valid_A.pkl', 'rb'))
		t_dict_B = pkl.load(open(self.database_path + '/train_only_B.pkl', 'rb'))
		tv_dict_B = pkl.load(open(self.database_path + '/train_and_valid_B.pkl', 'rb'))
		t_dict_C = pkl.load(open(self.database_path + '/train_only_C.pkl', 'rb'))
		tv_dict_C = pkl.load(open(self.database_path + '/train_and_valid_C.pkl', 'rb'))

		if total_number_folds == -1:
			total_number_folds = len(tvt_dict_A.keys())

		folds_A = []
		train_names_A = []
		valid_names_A = []
		test_names_A = []

		# Pretraining is done one time on one single fold (8/1/1 split -> number of k-folds = 10)
		folds_C, train_names_C, valid_names_C, test_names_C = build_one_fold(0, 10, t_dict_C, tv_dict_C, {}, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks, self.random_inst)

		# Pretraining is done one time on one single fold (8/1/1 split -> number of k-folds = 10)
		folds_B, train_names_B, valid_names_B, test_names_B = build_one_fold(0, 10, t_dict_B, tv_dict_B, {}, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks, self.random_inst)

		# Build the list of split_matrices
		for current_fold_number in range(total_number_folds):
			one_fold, this_train_names, this_valid_names, this_test_names = build_one_fold(current_fold_number, total_number_folds, t_dict_A, {}, tvt_dict_A, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks, self.random_inst)
			folds_A.append(one_fold)
			train_names_A.append(this_train_names)
			valid_names_A.append(this_valid_names)
			test_names_A.append(this_test_names)

		return folds_A, train_names_A, valid_names_A, test_names_A, folds_B, train_names_B, valid_names_B, test_names_B, folds_C, train_names_C, valid_names_C, test_names_C

	def get_folds(self, parameters, model_params):
		# Sadly, with the split of the database, shuffling the files can only be performed inside a same split
		self.logger.info('##### Building folds')
		# Load data and build K_folds
		time_load_0 = time.time()
		# K_folds[fold_index]['train','test' or 'valid'][index split]['batches' : [[234,14,54..],[..],[..]], 'matrices_path':[path_0,path_1,..]]
		if self.num_k_folds == 0:
			# this_K_folds, this_valid_names, this_test_names = build_folds(tracks_start_end, piano, orch, 10, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], RANDOM_SEED_FOLDS, logger_load=None)
			folds_A, train_names_A, valid_names_A, test_names_A, self.K_folds_B, self.train_names_B, self.valid_names_B, self.test_names_B, self.K_folds_C, self.train_names_C, self.valid_names_C, self.test_names_C = \
				self.__build_folds(10, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], parameters["num_max_contiguous_blocks"])
			self.K_folds_A = [folds_A[0]]
			self.train_names_A = [train_names_A[0]]
			self.valid_names_A = [valid_names_A[0]]
			self.test_names_A = [test_names_A[0]]
		elif self.num_k_folds == -1:
			raise Exception("num_k_folds = -1 Doesn't really make sense here")
		else:
			self.K_folds_A, self.train_names_A, self.valid_names_A, self.test_names_A, self.K_folds_B, self.train_names_B, self.valid_names_B, self.test_names_B, self.K_folds_C, self.train_names_C, self.valid_names_C, self.test_names_C \
				= self.__build_folds(self.num_k_folds, model_params["temporal_order"], parameters["batch_size"], parameters["long_range"], parameters["num_max_contiguous_blocks"])
		time_load = time.time() - time_load_0
		self.logger.info('TTT : Building folds took {} seconds'.format(time_load))
		return

	def submit_jobs(self, parameters, model_params, dimensions, track_paths_generation, save_bool, generate_bool, local):
		# Pretraning C
		config_folder_C = self.config_folder + "/stepC"
		if os.path.isdir(config_folder_C):
			shutil.rmtree(config_folder_C)
		os.mkdir(config_folder_C)
		# Submit worker
		submit_job(config_folder_C, parameters, model_params, dimensions, self.K_folds_C, 
			self.train_names_C, self.valid_names_C, self.test_names_C,
			track_paths_generation, True, False, local, self.logger)
		# Define new pre-trained model
		parameters['pretrained_model'] = os.path.join(config_folder_C, 'model_accuracy')		

		# Pretraning B
		config_folder_B = self.config_folder + "/stepB"
		if os.path.isdir(config_folder_B):
			shutil.rmtree(config_folder_B)
		os.mkdir(config_folder_B)
		# Submit worker
		submit_job(config_folder_B, parameters, model_params, dimensions, self.K_folds_B, 
			self.train_names_B, self.valid_names_B, self.test_names_B,
			track_paths_generation, True, False, local, self.logger)
		# Define new pre-trained model		
		parameters['pretrained_model'] = os.path.join(config_folder_B, 'model_accuracy')

		for K_fold_ind, K_fold in enumerate(self.K_folds_A):
			# Create fold folder
			config_folder_fold = self.config_folder + "/" + str(K_fold_ind)
			if os.path.isdir(config_folder_fold):
				shutil.rmtree(config_folder_fold)
			os.mkdir(config_folder_fold)
			# Submit worker
			submit_job(config_folder_fold, parameters, model_params, dimensions, K_fold, 
				self.train_names_A[K_fold_ind], self.valid_names_A[K_fold_ind], self.test_names_A[K_fold_ind],
				track_paths_generation, save_bool, generate_bool, local, self.logger)
		return