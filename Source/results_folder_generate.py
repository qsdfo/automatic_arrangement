#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import cPickle as pkl
import time
# Load data
from load_data import load_data_test
from generate import generate

def generate_midi(config_folder, generation_length, seed_size, quantization_write, logger_generate):
    ############################################################
    # Load the model and config
    ############################################################
    model_path = config_folder + '/model.pkl'
    model = pkl.load(open(model_path, 'rb'))
    param_path = config_folder + '/config.pkl'
    space = pkl.load(open(param_path, 'rb'))
    model_param = space['model']
    script_param = space['script']
    metadata_path = script_param['data_folder'] + '/metadata.pkl'

    if quantization_write is None:
        quantization_write = script_param['quantization']

    ############################################################
    # Create generation folder
    ############################################################
    generated_folder = config_folder + '/generated_sequences'
    if not os.path.isdir(generated_folder):
        os.makedirs(generated_folder)

    ############################################################
    # Load data
    ############################################################
    time_load_0 = time.time()
    piano_test, orchestra_test, _, generation_index \
        = load_data_test(script_param['data_folder'],
                         model_param['temporal_order'],
                         model_param['batch_size'],
                         binary_unit=script_param['binary_unit'],
                         skip_sample=script_param['skip_sample'],
                         logger_load=logger_generate)
    time_load_1 = time.time()
    logger_generate.info('TTT : Loading data took {} seconds'.format(time_load_1-time_load_0))

    ############################################################
    # Generate
    ############################################################
    time_generate_0 = time.time()
    generate(model,
             piano_test, orchestra_test, generation_index, metadata_path,
             generation_length, seed_size, quantization_write,
             generated_folder, logger_generate)
    time_generate_1 = time.time()
    logger_generate.info('TTT : Generating data took {} seconds'.format(time_generate_1-time_generate_0))
