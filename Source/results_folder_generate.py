#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

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

import reconstruct_pr
from acidano.utils.init import shared_zeros
from acidano.data_processing.midi.write_midi import write_midi


def generate_midi(config_folder, data_folder, generation_length, corruption_flag, logger_generate):
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

    ############################################################
    # Load data
    ############################################################
    time_load_0 = time.time()
    piano_checksum = model.checksum_database['piano_test']
    orchestra_checksum = model.checksum_database['orchestra_test']
    piano_test, orchestra_test, _, generation_index \
        = load_data_test(data_folder, 
                         piano_checksum, 
                         orchestra_checksum, 
                         temporal_order=model_param['temporal_order'], 
                         batch_size=model_param['batch_size'], 
                         skip_sample=script_param['skip_sample'],
                         avoid_silence=True, 
                         logger_load=None, 
                         generation_length=generation_length)
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

    generate(model, piano_test, orchestra_test, generation_index, model_param['temporal_order'])
    time_generate_1 = time.time()
    logger_generate.info('TTT : Generating data took {} seconds'.format(time_generate_1-time_generate_0))


def generate_midi_full_track_reference(config_folder, data_folder, track_path, seed_size, number_of_version, logger_generate):
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

    ############################################################
    # Read piano midi file
    ############################################################
    pr_piano, event_piano, non_silent_piano, instru_piano, name_piano, pr_orch, event_orch, non_silent_orch, instru_orch, name_orch, duration =\
        build_data_aux.process_folder(track_path, script_param['quantization'], script_param['unit_type'], script_param['temporal_granularity'], gapopen=3, gapextend=1)

    # Get the instrument mapping used for training
    metadata = pkl.load(open(metadata_path, 'rb'))
    instru_mapping = metadata['instru_mapping']

    # Instanciate piano pianoroll
    N_piano = instru_mapping['piano']['index_max']
    pr_piano_gen = np.zeros((duration, N_piano), dtype=np.float32)
    pr_piano_gen = build_data_aux.cast_small_pr_into_big_pr(pr_piano, {}, 0, duration, instru_mapping, pr_piano_gen)

    # Instanciate orchestra pianoroll with orchestra seed
    orch_seed_beginning = {k: v[:seed_size] for k, v in pr_orch.iteritems()}
    N_orchestra = metadata['N_orchestra']
    pr_orchestra_gen = np.zeros((duration, N_orchestra), dtype=np.float32)
    pr_orchestra_gen = build_data_aux.cast_small_pr_into_big_pr(orch_seed_beginning, instru_orch, 0, seed_size, instru_mapping, pr_orchestra_gen)
    pr_orchestra_truth = np.zeros((duration, N_orchestra), dtype=np.float32)
    pr_orchestra_truth = build_data_aux.cast_small_pr_into_big_pr(pr_orch, instru_orch, 0, duration, instru_mapping, pr_orchestra_truth)

    ############################################################
    ############################################################
    ############################################################
    # from acidano.visualization.numpy_array.visualize_numpy import visualize_mat, visualize_dict
    # save_folder_name = 'DEBUG'
    # visualize_dict(pr_orch_event, save_folder_name, 'orch_event', time_indices=(0,100))
    # visualize_dict(pr_orch_aligned, save_folder_name, 'orch_aligned', time_indices=(0,100))
    # visualize_mat(pr_orchestra[:100], save_folder_name, 'orch', time_indices=(0,100))
    # visualize_dict(pr_piano_event, save_folder_name, 'piano_event', time_indices=(0,100))
    # visualize_dict(pr_piano_aligned, save_folder_name, 'piano_aligned', time_indices=(0,100))
    # visualize_mat(pr_piano[:100], save_folder_name, 'piano', time_indices=(0,100))
    # import pdb; pdb.set_trace()
    ############################################################
    ############################################################
    ############################################################

    # Push them on the GPU
    pr_piano_shared = theano.shared(pr_piano_gen, name='piano_generation', borrow=True)
    pr_orchestra_shared = theano.shared(pr_orchestra_gen, name='orchestra_generation', borrow=True)

    # generation_index is the last index of the track we want to generate
    # We feed several time the same index to get different proposition of orchestration
    generation_index = np.asarray([duration-1, ] * number_of_version, dtype=np.int32)

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
    generated_sequences = generate(model, pr_piano_shared, pr_orchestra_shared, generation_index, seed_size)
    time_generate_1 = time.time()
    logger_generate.info('TTT : Generating data took {} seconds'.format(time_generate_1-time_generate_0))

    ############################################################
    # Reconstruct and write
    ############################################################
    for write_counter in xrange(generated_sequences.shape[0]):
        # Reconstruct
        pr_orchestra_clean = reconstruct_pr.time_reconstruction(generated_sequences[write_counter], non_silent_piano, event_piano)
        pr_orchestra = reconstruct_pr.instrument_reconstruction(pr_orchestra_clean, instru_mapping)
        # Write
        write_path = generated_folder + '/' + str(write_counter) + '_generated.mid'
        write_midi(pr_orchestra, script_param['quantization'], write_path, tempo=80)

    ############################################################
    ############################################################
    # Write original orchestration and piano scores, but reconstructed version, just to check
    A = reconstruct_pr.time_reconstruction(pr_piano_gen, non_silent_piano, event_piano)
    piano_reconstructed = reconstruct_pr.instrument_reconstruction_piano(A, instru_mapping)
    write_path = generated_folder + '/piano_reconstructed.mid'
    write_midi(piano_reconstructed, script_param['quantization'], write_path, tempo=80)
    #
    A = reconstruct_pr.time_reconstruction(pr_orchestra_truth, non_silent_piano, event_piano)
    orchestra_reconstructed = reconstruct_pr.instrument_reconstruction(A, instru_mapping)
    write_path = generated_folder + '/orchestra_reconstructed.mid'
    write_midi(orchestra_reconstructed, script_param['quantization'], write_path, tempo=80)
    ############################################################
    ############################################################
