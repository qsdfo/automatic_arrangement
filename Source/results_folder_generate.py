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

import acidano.data_processing.utils.pianoroll_processing as pianoroll_processing
import acidano.data_processing.utils.event_level as event_level
import acidano.data_processing.utils.time_warping as time_warping
import theano
import re

from acidano.utils.init import shared_zeros
import acidano.data_processing.utils.unit_type as Unit_type


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
             generation_length, seed_size, script_param["quantization"], script_param['temporal_granularity'], None,
             generated_folder, logger_generate)
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
    # We don't want to process alignment here !!
    # Just compute the new event of the piano, and orchestrate the corresponding frames
    pr0, instru0, _, name0, pr1, instru1, _, name1 = build_data_aux.get_instru_and_pr_from_folder_path(track_path, script_param['quantization'])

    # Find which pr is piano which one is orchestra
    pr_piano_dict, instru_piano, name_piano, pr_orch_dict, instru_orch, name_orch =\
        build_data_aux.discriminate_between_piano_and_orchestra(pr0, instru0, name0, pr1, instru1, name1)

    # Unit type
    pr_piano_dict = Unit_type.from_rawpr_to_type(pr_piano_dict, script_param['unit_type'])
    pr_orch_dict = Unit_type.from_rawpr_to_type(pr_orch_dict, script_param['unit_type'])

    # Temporal granularity
    if script_param['temporal_granularity'] == 'event_level':
        event_ind_piano = event_level.get_event_ind_dict(pr_piano_dict)
        pr_piano_event = time_warping.warp_pr_aux(pr_piano_dict, event_ind_piano)
        event_ind_orch = event_level.get_event_ind_dict(pr_orch_dict)
        pr_orch_event = time_warping.warp_pr_aux(pr_orch_dict, event_ind_orch)

    # Align tracks
    pr_piano_aligned, trace_piano, pr_orch_aligned, trace_orch, trace_prod, duration =\
        build_data_aux.align_tracks(pr_piano_event, pr_orch_event, script_param['unit_type'], gapopen=3, gapextend=1)

    # Get seed_size frames of the aligned pr
    piano_seed_beginning = {k: v[:seed_size] for k, v in pr_piano_aligned.iteritems()}
    orch_seed_beginning = {k: v[:seed_size] for k, v in pr_orch_aligned.iteritems()}
    # Get event_part
    start_ind_event, list_seed_event = get_event_seed_reconstruction(trace_piano, trace_prod, seed_size)
    piano_end = {k: v[start_ind_event:] for k, v in pr_piano_event.iteritems()}

    # Indices for the reconstruction
    event_ind_reconstruction = np.concatenate([event_ind_piano[list_seed_event], event_ind_piano[start_ind_event:]])

    # Get length of pr
    duration_end = pianoroll_processing.get_pianoroll_time(piano_end)
    duration = seed_size + duration_end

    # Get the instrument mapping used for training
    metadata = pkl.load(open(metadata_path, 'rb'))
    instru_mapping = metadata['instru_mapping']

    # Instanciate piano pianoroll
    N_piano = instru_mapping['piano']['index_max']
    pr_piano = np.zeros((duration, N_piano), dtype=np.float32)
    pr_piano = build_data_aux.cast_small_pr_into_big_pr(piano_seed_beginning, {}, 0, seed_size, instru_mapping, pr_piano)
    pr_piano = build_data_aux.cast_small_pr_into_big_pr(piano_end, {}, seed_size, duration_end, instru_mapping, pr_piano)

    # Instanciate orchestra pianoroll with orchestra seed
    N_orchestra = metadata['N_orchestra']
    pr_orchestra = np.zeros((duration, N_orchestra), dtype=np.float32)
    pr_orchestra = build_data_aux.cast_small_pr_into_big_pr(orch_seed_beginning, instru_orch, 0, seed_size, instru_mapping, pr_orchestra)

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
    ############################################################

    # Push them on the GPU
    pr_piano_shared = theano.shared(pr_piano, name='piano_generation', borrow=True)
    pr_orchestra_shared = theano.shared(pr_orchestra, name='orchestra_generation', borrow=True)

    # Generation parameters
    generation_length = duration
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
    generate(model,
             pr_piano_shared, pr_orchestra_shared, generation_index, metadata_path,
             generation_length, seed_size, script_param["quantization"], script_param['temporal_granularity'], event_ind_reconstruction,
             generated_folder, logger_generate)
    time_generate_1 = time.time()
    logger_generate.info('TTT : Generating data took {} seconds'.format(time_generate_1-time_generate_0))


def get_event_seed_reconstruction(trace_piano, trace_prod, seed_size):
    counter = 0
    counter_prod = 0
    list_ind = []
    sum_trace = [a+b for a, b in zip(trace_piano, trace_prod)]
    for ind in sum_trace:
        if counter_prod == seed_size:
            break
        if ind == 2:
            list_ind.append(counter)
            counter += 1
            counter_prod += 1
        elif ind == 1:
            counter += 1
    return counter, list_ind
