#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

#################################################
#################################################
#################################################
# Note :
#   - pitch range for each instrument is set based on the observed pitch range of the database
#   - for test set, we picked seminal examples. See the name_db_{test;train;valid}.txt files that
#       list the files path :
#           - beethoven/liszt symph 5 - 1 : liszt_classical_archive/16
#           - mouss/ravel pictures exhib : bouliane/22
#################################################
#################################################
#################################################

import os
import glob
import shutil
import re
import numpy as np
import LOP.Scripts.config as config
import build_data_aux
import build_data_aux_no_piano
import pickle as pkl
import avoid_tracks 
# memory issues
import gc
import sys

import LOP.Scripts.config as config

DEBUG=True


def update_instru_mapping(folder_path, instru_mapping, T, quantization):
	logging.info(folder_path)
	if not os.path.isdir(folder_path):
		return instru_mapping, T
	
	# Is there an original piano score or do we have to create it ?
	num_music_file = max(len(glob.glob(folder_path + '/*.mid')), len(glob.glob(folder_path + '/*.xml')))
	if num_music_file == 2:
		is_piano = True
	elif num_music_file == 1:
		is_piano = False
	else:
		raise Exception("CAVAVAVAMAVAL")

	# Read pr
	if is_piano:
		pr_piano, _, _, instru_piano, _, pr_orch, _, _, instru_orch, _, duration =\
			build_data_aux.process_folder(folder_path, quantization, temporal_granularity, gapopen=3, gapextend=1)
	else:
		try:
			pr_piano, _, _, instru_piano, _, pr_orch, _, _, instru_orch, _, duration =\
				build_data_aux_no_piano.process_folder_NP(folder_path, quantization, temporal_granularity)
		except:
			duration=None
			logging.warning("Could not read file in " + folder_path)
	
	# if len(set(instru_orch.values())) < 4:
	#     import pdb; pdb.set_trace()

	if duration is None:
		# Files that could not be aligned
		return instru_mapping, T
	
	T += duration
	
	# Modify the mapping from instrument to indices in pianorolls and pitch bounds
	instru_mapping = build_data_aux.instru_pitch_range(instrumentation=instru_piano,
													   pr=pr_piano,
													   instru_mapping=instru_mapping,
													   )
	# remark : instru_mapping would be modified if it is only passed to the function,
	#                   f(a)  where a is modified inside the function
	# but i prefer to make the reallocation explicit
	#                   a = f(a) with f returning the modified value of a.
	# Does it change anything for computation speed ? (Python pass by reference,
	# but a slightly different version of it, not clear to me)
	instru_mapping = build_data_aux.instru_pitch_range(instrumentation=instru_orch,
													   pr=pr_orch,
													   instru_mapping=instru_mapping,
													   )

	return instru_mapping, T


def get_dim_matrix(folder_paths, folder_paths_pretraining, meta_info_path, quantization, temporal_granularity, T_limit, logging=None):
	logging.info("##########")
	logging.info("Get dimension informations")
	# Determine the temporal size of the matrices
	# If the two files have different sizes, we use the shortest (to limit the use of memory,
	# we better contract files instead of expanding them).
	# Get instrument names
	instru_mapping = {}
	# instru_mapping = {'piano': {'pitch_min': 24, 'pitch_max':117, 'ind_min': 0, 'ind_max': 92},
	#                         'harp' ... }
	folder_paths_splits = {}
	folder_paths_pretraining_splits = {}

	##########################################################################################
	# Pre-train        
	split_counter = 0
	T_pretraining = 0
	folder_paths_pretraining_split = []
	for folder_path_pre in folder_paths_pretraining:
		if T_pretraining>T_limit:
			folder_paths_pretraining_splits[split_counter] = (T_pretraining, folder_paths_pretraining_split)
			T_pretraining = 0
			folder_paths_pretraining_split = []
			split_counter+=1
		folder_path_pre = folder_path_pre.rstrip()
		instru_mapping, T_pretraining = update_instru_mapping(folder_path_pre, instru_mapping, T_pretraining, quantization)
		folder_paths_pretraining_split.append(folder_path_pre)
	if len(folder_paths_pretraining) > 0:
		# Don't forget the last one !
		folder_paths_pretraining_splits[split_counter] = (T_pretraining, folder_paths_pretraining_split)
	##########################################################################################

	##########################################################################################
	# Train
	split_counter = 0
	T = 0
	folder_paths_split = []
	for folder_path in folder_paths:
		if T>T_limit:
			folder_paths_splits[split_counter] = (T, folder_paths_split)
			T = 0
			folder_paths_split = []
			split_counter+=1
		folder_path = folder_path.rstrip()
		instru_mapping, T = update_instru_mapping(folder_path, instru_mapping, T, quantization)
		folder_paths_split.append(folder_path)
	# Don't forget the last one !
	if len(folder_paths) > 0:
		folder_paths_splits[split_counter] = (T, folder_paths_split)
	##########################################################################################
	
	##########################################################################################
	# Build the index_min and index_max in the instru_mapping dictionary
	counter = 0
	for k, v in instru_mapping.items():
		if k == 'Piano':
			index_min = 0
			index_max = v['pitch_max'] - v['pitch_min']
			v['index_min'] = index_min
			v['index_max'] = index_max
			continue
		index_min = counter
		counter = counter + v['pitch_max'] - v['pitch_min']
		index_max = counter
		v['index_min'] = index_min
		v['index_max'] = index_max

	# Instanciate the matrices
	temp = {}
	temp['instru_mapping'] = instru_mapping
	temp['quantization'] = quantization
	temp['folder_paths_splits'] = folder_paths_splits
	temp['folder_paths_pretraining_splits'] = folder_paths_pretraining_splits
	temp['N_orchestra'] = counter
	pkl.dump(temp, open(meta_info_path, 'wb'))
	##########################################################################################

	return

def build_split_matrices(folder_paths, destination_folder, chunk_size, instru_mapping, N_piano, N_orchestra):
	file_counter = 0
	train_only_files={}
	train_and_valid_files={}

	for folder_path in folder_paths:
		#############
		# Read file
		folder_path = folder_path.rstrip()
		logging.info(" : " + folder_path)
		if not os.path.isdir(folder_path):
			continue

		if folder_path in avoid_tracks.no_valid_tracks():
			train_only_files[folder_path] = []
		else:
			train_and_valid_files[folder_path] = []

		# Is there an original piano score or do we have to create it ?
		num_music_file = max(len(glob.glob(folder_path + '/*.mid')), len(glob.glob(folder_path + '/*.xml')))
		if num_music_file == 2:
			is_piano = True
		elif num_music_file == 1:
			is_piano = False
		else:
			raise Exception("CAVAVAVAMAVAL")

		# Get pr, warped and duration
		if is_piano:
			new_pr_piano, _, new_duration_piano, _, new_name_piano, new_pr_orchestra, _, new_duration_orch, new_instru_orchestra, _, duration\
				= build_data_aux.process_folder(folder_path, quantization, temporal_granularity, gapopen=3, gapextend=1)
		else:
			try:
				new_pr_piano, _, new_duration_piano, _, new_name_piano, new_pr_orchestra, _, new_duration_orch, new_instru_orchestra, _, duration\
					= build_data_aux_no_piano.process_folder_NP(folder_path, quantization, temporal_granularity)
			except:
				logging.warning("Could not read file in " + folder_path)
				continue

		# Skip shitty files
		if new_pr_piano is None:
			# It's definitely not a match...
			# Check for the files : are they really a piano score and its orchestration ??
			with(open('log_build_db.txt', 'a')) as f:
				f.write(folder_path + '\n')
			continue

		pr_orch = build_data_aux.cast_small_pr_into_big_pr(new_pr_orchestra, new_instru_orchestra, 0, duration, instru_mapping, np.zeros((duration, N_orchestra)))
		pr_piano = build_data_aux.cast_small_pr_into_big_pr(new_pr_piano, {}, 0, duration, instru_mapping, np.zeros((duration, N_piano)))

		# Small section for generating piano only midi files (pour Mathieu, train embeddings)
		# new_file_name = re.split('/', new_name_piano)[-1]
		# try:
		# 	write_midi(new_pr_piano, 1000, "Piano_files_for_embeddings/" + new_file_name, tempo=80)
		# except:
		# 	logging.warning("Failed writing")

		#############
		# Split
		last_index = pr_piano.shape[0]
		start_indices = range(0, pr_piano.shape[0], chunk_size)

		for split_counter, start_index in enumerate(start_indices):
			this_split_folder = destination_folder + '/' + str(file_counter) + '_' + str(split_counter)
			os.mkdir(this_split_folder)
			end_index = min(start_index + chunk_size, last_index)
		
			section = pr_piano[start_index: end_index]
			section_cast = section.astype(np.float32)
			np.save(this_split_folder + '/pr_piano.npy', section_cast)

			section = pr_orch[start_index: end_index]
			section_cast = section.astype(np.float32)
			np.save(this_split_folder + '/pr_orch.npy', section_cast)

			section = new_duration_piano[start_index: end_index]
			section_cast = np.asarray(section, dtype=np.int8)
			np.save(this_split_folder + '/duration_piano.npy', section_cast)

			section = new_duration_orch[start_index: end_index]
			section_cast = np.asarray(section, dtype=np.int8)
			np.save(this_split_folder + '/duration_orch.npy', section_cast)

			# Keep track of splits
			if folder_path in avoid_tracks.no_valid_tracks():
				train_only_files[folder_path].append(this_split_folder)
			else:
				train_and_valid_files[folder_path].append(this_split_folder)

		file_counter+=1

	return train_and_valid_files, train_only_files

def build_data(folder_paths, folder_paths_pretraining, meta_info_path, quantization, temporal_granularity, store_folder, logging=None):

	# Get dimensions
	if DEBUG:
		T_limit = 20000
	else:
		T_limit = 1e6
	
	get_dim_matrix(folder_paths, folder_paths_pretraining, meta_info_path=meta_info_path, quantization=quantization, temporal_granularity=temporal_granularity, T_limit=T_limit, logging=logging)

	logging.info("##########")
	logging.info("Build data")

	statistics = {}
	statistics_pretraining = {}

	temp = pkl.load(open(meta_info_path, 'rb'))
	instru_mapping = temp['instru_mapping']
	quantization = temp['quantization']
	N_orchestra = temp['N_orchestra']
	N_piano = instru_mapping['Piano']['index_max']

	# Build the pitch and instru indicator vectors
	# We use integer to identify pitches and instrument
	# Used for NADE rule-based masking, not for reconstruction
	pitch_orch = np.zeros((N_orchestra), dtype="int8")-1
	instru_orch = np.zeros((N_orchestra), dtype="int8")-1
	counter = 0
	for k, v in instru_mapping.items():
		if k == "Piano":
			continue
		pitch_orch[v['index_min']:v['index_max']] = np.arange(v['pitch_min'], v['pitch_max']) % 12
		instru_orch[v['index_min']:v['index_max']] = counter
		counter += 1
	pitch_piano = np.arange(instru_mapping['Piano']['pitch_min'], instru_mapping['Piano']['pitch_max'], dtype='int8') % 12
	np.save(store_folder + '/pitch_orch.npy', pitch_orch)
	np.save(store_folder + '/instru_orch.npy', instru_orch)
	np.save(store_folder + '/pitch_piano.npy', pitch_piano)

	###################################################################################################
	# Build matrices
	chunk_size = config.build_parameters()["chunk_size"]
	training_split_folder = os.path.join(store_folder, "split_matrices")
	os.mkdir(training_split_folder)
	pretraining_split_folder = os.path.join(store_folder, "split_matrices_pretraining")
	os.mkdir(pretraining_split_folder)

	train_and_valid_files, train_only_files = build_split_matrices(folder_paths, training_split_folder, chunk_size, instru_mapping, N_piano, N_orchestra)
	pre_train_and_valid_files, pre_train_only_files = build_split_matrices(folder_paths_pretraining, pretraining_split_folder, chunk_size, instru_mapping, N_piano, N_orchestra)

	# Save files' lists
	pkl.dump(train_and_valid_files, open(store_folder + '/train_and_valid_files.pkl', 'wb'))
	pkl.dump(train_only_files, open(store_folder + '/train_only_files.pkl', 'wb'))
	pkl.dump(pre_train_and_valid_files, open(store_folder + '/train_and_valid_files_pretraining.pkl', 'wb'))
	pkl.dump(pre_train_only_files, open(store_folder + '/train_only_files_pretraining.pkl', 'wb'))

	metadata = {}
	metadata['quantization'] = quantization
	metadata['N_orchestra'] = N_orchestra
	metadata['N_piano'] = N_piano
	metadata['chunk_size'] = chunk_size
	metadata['instru_mapping'] = instru_mapping
	metadata['quantization'] = quantization
	metadata['temporal_granularity'] = temporal_granularity
	metadata['store_folder'] = store_folder
	with open(store_folder + '/metadata.pkl', 'wb') as outfile:
		pkl.dump(metadata, outfile)
	return

if __name__ == '__main__':
	import logging
	# log file
	log_file_path = 'log_build_data'
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

	# Set up
	# NOTE : can't do data augmentation with K-folds, or it would require to build K times the database
	# because train is data augmented but not test and validate
	temporal_granularity='event_level'
	quantization=8
	pretraining_bool=False

	# Database have to be built jointly so that the ranges match
	DATABASE_PATH = config.database_root()
	DATABASE_PATH_PRETRAINING = config.database_pretraining_root()
	
	if DEBUG:
		DATABASE_NAMES = [DATABASE_PATH + "/debug"] #, "imslp"]
	else:
		DATABASE_NAMES = [
			DATABASE_PATH + "/bouliane", 
			DATABASE_PATH + "/hand_picked_Spotify", 
			DATABASE_PATH + "/liszt_classical_archives", 
			DATABASE_PATH + "/imslp"
			# DATABASE_PATH_PRETRAINING + "/OpenMusicScores",
			# DATABASE_PATH_PRETRAINING + "/Kunstderfuge", 
			# DATABASE_PATH_PRETRAINING + "/Musicalion", 
			# DATABASE_PATH_PRETRAINING + "/Mutopia"
		]
	
	if DEBUG:
		DATABASE_NAMES_PRETRAINING = [DATABASE_PATH_PRETRAINING + "/debug"]
	else:
		DATABASE_NAMES_PRETRAINING = [
			DATABASE_PATH_PRETRAINING + "/OpenMusicScores",
			DATABASE_PATH_PRETRAINING + "/Kunstderfuge", 
			DATABASE_PATH_PRETRAINING + "/Musicalion", 
			DATABASE_PATH_PRETRAINING + "/Mutopia"
		]

	data_folder = config.data_root() + '/Data'
	if DEBUG:
		data_folder += '_DEBUG'
	if pretraining_bool:
		data_folder += '_pretraining'
	data_folder += '_tempGran' + str(quantization)

	# data_folder += '_pretraining_only'
	
	if os.path.isdir(data_folder):
		shutil.rmtree(data_folder)
	os.makedirs(data_folder)

	# Create a list of paths
	def build_filepaths_list(path):
		folder_paths = []
		for file_name in os.listdir(path):
			if file_name != '.DS_Store':
				this_path = os.path.join(path, file_name)
				folder_paths.append(this_path)
		return folder_paths

	folder_paths = []
	for path in DATABASE_NAMES:
		folder_paths += build_filepaths_list(path)

	folder_paths_pretraining = []
	if pretraining_bool:
		for path in DATABASE_NAMES_PRETRAINING:
			folder_paths_pretraining += build_filepaths_list(path)

	# Remove garbage tracks
	avoid_tracks_list = avoid_tracks.avoid_tracks()
	folder_paths = [e for e in folder_paths if e not in avoid_tracks_list]
	folder_paths_pretraining = [e for e in folder_paths_pretraining if e not in avoid_tracks_list]

	print("Training : " + str(len(folder_paths)))
	print("Pretraining : " + str(len(folder_paths_pretraining)))

	build_data(folder_paths=folder_paths,
			   folder_paths_pretraining=folder_paths_pretraining,
			   meta_info_path=data_folder + '/temp.pkl',
			   quantization=quantization,
			   temporal_granularity=temporal_granularity,
			   store_folder=data_folder,
			   logging=logging)
