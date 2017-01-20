#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import cPickle as pkl
import time
# Load data
from load_data import load_data_test
from generate import generate
import build_data_aux
import numpy as np

import theano
import re

from acidano.utils.init import shared_zeros

def generate_midi(config_folder, data_folder, generation_length, seed_size, quantization_write, corruption_flag, logger_generate):
    if logger_generate is None:
        import logging
        logging.basicConfig(level=logging.WARNING)
        logger_generate = logging.getLogger('generate')
    ############################################################
    # Load the model and config
    ############################################################
    model_path = config_folder + '/model.pkl'
    model = pkl.load(open(model_path, 'rb'))
    param_path = config_folder + '/config.pkl'
    space = pkl.load(open(param_path, 'rb'))
    model_param = space['model']
    script_param = space['script']
    metadata_path = data_folder + '/metadata.pkl'

    if quantization_write is None:
        quantization_write = script_param['quantization']

    ############################################################
    # Load data
    ############################################################
    time_load_0 = time.time()
    piano_checksum = model.checksum_database['piano_test']
    orchestra_checksum = model.checksum_database['orchestra_test']
    piano_test, orchestra_test, _, generation_index \
        = load_data_test(data_folder,
                         piano_checksum, orchestra_checksum,
                         model_param['temporal_order'],
                         model_param['batch_size'],
                         binary_unit=script_param['binary_unit'],
                         skip_sample=script_param['skip_sample'],
                         logger_load=logger_generate)
    time_load_1 = time.time()
    logger_generate.info('TTT : Loading data took {} seconds'.format(time_load_1-time_load_0))

    ############################################################
    # Corruption and create generation folder
    ############################################################
    generated_folder = config_folder + '/generated_sequences'
    if 'piano' == corruption_flag:
        piano_test = shared_zeros(piano_test.get_value(borrow=True).shape)
        generated_folder = generated_folder + '_corrupted_piano'
        logger_generate.info("CORRUPTED PIANO :\nGenerating data with piano set to O")
    elif 'orchestra' == corruption_flag:
        orchestra_test = shared_zeros(orchestra_test.get_value(borrow=True).shape)
        generated_folder = generated_folder + '_corrupted_orchestra'
        logger_generate.info("CORRUPTED ORCHESTRA :\nGenerating data with orchestra seed set to O")
    else:
        logger_generate.info("NO CORRUPTION OF THE DATA :\nnormal generation")
    if not os.path.isdir(generated_folder):
        os.makedirs(generated_folder)

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

def generate_midi_full_track_reference(config_folder, data_folder, track_path, seed_size, quantization_write, number_of_version, logger_generate):
    # This function generate the orchestration of a full track
    if logger_generate is None:
        import logging
        logging.basicConfig(level=logging.WARNING)
        logger_generate = logging.getLogger('generate')
    ############################################################
    # Load the model and config
    ############################################################
    model_path = config_folder + '/model.pkl'
    model = pkl.load(open(model_path, 'rb'))
    param_path = config_folder + '/config.pkl'
    space = pkl.load(open(param_path, 'rb'))
    script_param = space['script']
    metadata_path = data_folder + '/metadata.pkl'

    if quantization_write is None:
        quantization_write = script_param['quantization']

    ############################################################
    # Read piano midi file
    ############################################################
    # Get dictionnary representing the aligned midi tracks
    piano_test_dict, instru_piano, name_piano, orchestra_test_dict, instru_orchestra, name_orchestra, duration\
        = build_data_aux.process_folder(track_path, quantization_write, script_param['temporal_granularity'], logger_generate)

    # Get the instrument mapping used for training
    metadata = pkl.load(open(metadata_path, 'rb'))
    instru_mapping = metadata['instru_mapping']

    # Instanciate track pianoroll
    N_orchestra = metadata['N_orchestra']
    N_piano = instru_mapping['piano']['index_max']
    pr_orchestra = np.zeros((duration, N_orchestra), dtype=np.float32)
    pr_piano = np.zeros((duration, N_piano), dtype=np.float32)

    # Fill them with the dictionnary representation of the track, respecting the instrument mapping
    pr_piano = build_data_aux.cast_small_pr_into_big_pr(piano_test_dict, {}, 0, duration, instru_mapping, pr_piano)
    pr_orchestra = build_data_aux.cast_small_pr_into_big_pr(orchestra_test_dict, instru_orchestra, 0, duration, instru_mapping, pr_orchestra)

    # Is it binary units ?
    binary_unit = script_param['binary_unit']
    if binary_unit:
        pr_piano[np.nonzero(pr_piano)] = 1
        pr_orchestra[np.nonzero(pr_orchestra)] = 1
    else:
        # Much easier to work with unit between 0 and 1, for several reason :
        #       - 'normalized' values
        #       - same reconstruction as for binary units when building midi files
        pr_piano = pr_piano / 127
        pr_orchestra = pr_orchestra / 127

    # Push them on the GPU
    pr_piano_shared = theano.shared(pr_piano, name='piano_generation', borrow=True)
    pr_orchestra_shared = theano.shared(pr_orchestra, name='orchestra_generation', borrow=True)

    # Generation parameters
    generation_length = duration - 150
    seed_size = 20
    # generation_index is the last index of the track we want to generate
    # We feed several time the same index to get different proposition of orchestration
    generation_index = np.asarray([duration-1,] * number_of_version, dtype=np.int32)

    # Store folder
    string = re.split(r'/', name_piano)[-1]
    name_track = re.sub('piano_solo.mid', '', string)

    generated_folder = config_folder + '/generation_reference_example/' + name_track
    if not os.path.isdir(generated_folder):
        os.makedirs(generated_folder)

    ############################################################
    # Generate
    ############################################################
    time_generate_0 = time.time()
    generate(model,
             pr_piano_shared, pr_orchestra_shared, generation_index, metadata_path,
             generation_length, seed_size, quantization_write,
             generated_folder, logger_generate)
    time_generate_1 = time.time()
    logger_generate.info('TTT : Generating data took {} seconds'.format(time_generate_1-time_generate_0))
