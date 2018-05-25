#!/usr/bin/env pyth
# -*- coding: utf8 -*-

import subprocess
import os
import pickle as pkl
import train_wrapper


def submit_job(config_folder_fold, parameters, model_params, dimensions, K_fold, 
	train_names, valid_names, test_names,
	track_paths_generation, save_bool, generate_bool, local, logger):
	
	context_folder = config_folder_fold + '/context'
	os.mkdir(context_folder)

	# Save all the arguments of the wrapper script
	pkl.dump(parameters, open(context_folder + "/parameters.pkl", 'wb')) 
	pkl.dump(model_params, open(context_folder + '/model_params.pkl', 'wb'))
	pkl.dump(dimensions , open(context_folder + '/dimensions.pkl', 'wb'))
	pkl.dump(K_fold, open(context_folder + '/K_fold.pkl', 'wb'))
	pkl.dump(test_names, open(context_folder + '/test_names.pkl', 'wb'))
	pkl.dump(track_paths_generation, open(context_folder + '/track_paths_generation.pkl', 'wb'))
	pkl.dump(save_bool , open(context_folder + '/save_bool.pkl', 'wb'))
	pkl.dump(generate_bool , open(context_folder + '/generate_bool.pkl', 'wb'))
	# Write filenames of this split
	with open(os.path.join(config_folder_fold, "train_names.txt"), "w") as f:
		for filename in train_names:
			f.write(filename + "\n")
	with open(os.path.join(config_folder_fold, "test_names.txt"), "w") as f:
		for filename in test_names:
			f.write(filename + "\n")
	with open(os.path.join(config_folder_fold, "valid_names.txt"), "w") as f:
		for filename in valid_names:
			f.write(filename + "\n")

	if local:
		# subprocess.check_output('python train_wrapper.py ' + config_folder_fold, shell=True)
		train_wrapper.train_wrapper(parameters, model_params, 
			dimensions, config_folder_fold, K_fold,
			track_paths_generation, 
			save_bool, generate_bool, logger)
	else:	
		# Write pbs script
		file_pbs = context_folder + '/submit.pbs'

		split_config_folder_fold = re.split('/', config_folder_fold)
		script_name = split_config_folder_fold[-4] + "__" + split_config_folder_fold[-3] + "__" + split_config_folder_fold[-2] + "__" + split_config_folder_fold[-1]

		text_pbs = """#!/bin/bash

#PBS -j oe
#PBS -N job_outputs/""" + script_name + """
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=""" + str(parameters['walltime']) + """:00:00

module load foss/2015b
module load Tensorflow/1.0.0-Python-3.5.2
source ~/Virtualenvs/tf_3/bin/activate

SRC=/home/crestel/Source/automatic_arrangement/Source/LOP/Scripts
cd $SRC
python train_wrapper.py '""" + config_folder_fold + "'"

		with open(file_pbs, 'w') as f:
			f.write(text_pbs)

		job_id = subprocess.check_output('qsub ' + file_pbs, shell=True)
		
		return job_id