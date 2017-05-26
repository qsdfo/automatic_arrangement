#!/usr/bin/env python
# -*- coding: utf8 -*-

import time
import logging
import re
import os
import keras as K
import numpy as np
import cPickle as pkl
import glob

import build_data_aux
import reconstruct_pr

import acidano.data_processing.utils.unit_type as Unit_type
import acidano.data_processing.utils.pianoroll_processing as pianoroll_processing
import acidano.data_processing.utils.event_level as event_level
import acidano.data_processing.utils.time_warping as time_warping
from acidano.data_processing.midi.write_midi import write_midi
from assert_database import assert_database


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


def generate_keras(model_path, data_path, track_path):
    metadata_path = data_path + '/metadata.pkl'
    model = K.models.load_model(model_path + '/model.h5')
    config_path = model_path + '/config.pkl'
    ####################################################################
    ####################################################################
    # log file
    log_file_path = model_path + '/log_generate'
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
    ####################################################################
    ####################################################################

    ####################################################################
    ####################################################################
    logging.info('########################################################')
    logging.info('Generating...')
    quantization = 100
    unit_type = 'binary'
    temporal_granularity = 'event_level'
    seed_size = 20

    with open(config_path, 'rb') as f:
        config = pkl.load(f)
    script_param = config['script']
    model_param = config['model']

    # Get the instrument mapping used for training
    metadata = pkl.load(open(metadata_path, 'rb'))
    instru_mapping = metadata['instru_mapping']

    assert_database(metadata, script_param)

    ############################################################
    # Read piano midi file
    ############################################################
    # We don't want to process alignment here !!
    # Just compute the new event of the piano, and orchestrate the corresponding frames
    pr0, instru0, _, name0, pr1, instru1, _, name1 = build_data_aux.get_instru_and_pr_from_folder_path(track_path, quantization)

    # Find which pr is piano which one is orchestra
    pr_piano_dict, instru_piano, name_piano, pr_orch_dict, instru_orch, name_orch =\
        build_data_aux.discriminate_between_piano_and_orchestra(pr0, instru0, name0, pr1, instru1, name1)

    # Unit type
    pr_piano_dict = Unit_type.from_rawpr_to_type(pr_piano_dict, unit_type)
    pr_orch_dict = Unit_type.from_rawpr_to_type(pr_orch_dict, unit_type)

    # Temporal granularity
    if temporal_granularity == 'event_level':
        event_ind_piano = event_level.get_event_ind_dict(pr_piano_dict)
        pr_piano_dict = time_warping.warp_pr_aux(pr_piano_dict, event_ind_piano)
        event_ind_orch = event_level.get_event_ind_dict(pr_orch_dict)
        pr_orch_dict = time_warping.warp_pr_aux(pr_orch_dict, event_ind_orch)

    # Align tracks
    pr_piano_aligned, pr_orch_aligned, trace_prod, duration =\
        build_data_aux.align_tracks(pr_piano_dict, pr_orch_dict, unit_type, gapopen=3, gapextend=1)

    # Get seed_size frames of the aligned pr
    piano_seed_beginning = {k: v[:seed_size] for k, v in pr_piano_aligned.iteritems()}
    orch_seed_beginning = {k: v[:seed_size] for k, v in pr_orch_aligned.iteritems()}
    # Get event_part
    start_ind_event, list_seed_event = get_event_seed_reconstruction(trace_piano, trace_prod, seed_size)
    piano_end = {k: v[start_ind_event:] for k, v in pr_piano_dict.iteritems()}

    # Indices for the reconstruction
    event_ind_reconstruction = np.concatenate([event_ind_piano[list_seed_event], event_ind_piano[start_ind_event:]])

    # Get length of pr
    duration_end = pianoroll_processing.get_pianoroll_time(piano_end)
    duration = seed_size + duration_end

    # Instanciate piano pianoroll
    N_piano = instru_mapping['piano']['index_max']
    pr_piano = np.zeros((duration, N_piano), dtype=np.float32)
    pr_piano = build_data_aux.cast_small_pr_into_big_pr(piano_seed_beginning, {}, 0, seed_size, instru_mapping, pr_piano)
    pr_piano = build_data_aux.cast_small_pr_into_big_pr(piano_end, {}, seed_size, duration_end, instru_mapping, pr_piano)

    # Generation out of scratch
    # Instanciate orchestra pianoroll with zeros only
    N_orchestra = metadata['N_orchestra']
    pr_orchestra = np.zeros((duration, N_orchestra), dtype=np.float32)
    pr_orchestra = build_data_aux.cast_small_pr_into_big_pr(orch_seed_beginning, instru_orch, 0, seed_size, instru_mapping, pr_orchestra)

    # Store folder
    string = re.split(r'/', name_piano)[-1]
    name_track = re.sub(r'(_solo|_orch)', '', string)

    generated_folder = model_path + '/generation_reference_example/' + name_track
    if not os.path.isdir(generated_folder):
        os.makedirs(generated_folder)

    ############################################################
    # Generate
    ############################################################
    time_generate_0 = time.time()
    for t in range(seed_size, duration):
        orch_past = pr_orchestra[t-model_param['temporal_order']:t]
        orch_past_reshape = np.reshape(orch_past, (1, orch_past.shape[0], orch_past.shape[1]))
        piano_t = pr_piano[t]
        piano_t_reshape = np.reshape(piano_t, (1, piano_t.shape[0]))
        gen_frame_distrib = model.predict(x={'orch_seq': orch_past_reshape, 'piano_t': piano_t_reshape},
                                          batch_size=1)
        gen_frame = np.random.binomial(n=1, p=gen_frame_distrib)
        pr_orchestra[t] = gen_frame
    time_generate_1 = time.time()
    logging.info('TTT : Generating data took {} seconds'.format(time_generate_1-time_generate_0))

    # Write generated midi
    pr_orchestra = reconstruct_pr.reconstruct_pr(pr_orchestra, instru_mapping)
    if (temporal_granularity == 'event_level') and (event_ind_reconstruction is not None):
        # Place the generated vectors at the time indices in event_indices
        pr_orchestra = reconstruct_pr.place_event_level(pr_orchestra, event_ind_reconstruction)
        quantization_write = quantization
    elif temporal_granularity == "event_level":
        quantization_write = 1
    write_path = generated_folder + '/orchestra_generated.mid'
    write_midi(pr_orchestra, quantization_write, write_path, tempo=80)

    # Write original piano
    pr_piano_seed = reconstruct_pr.reconstruct_piano(pr_piano, instru_mapping)
    if (temporal_granularity == 'event_level') and (event_ind_reconstruction is not None):
        # Place the generated vectors at the time indices in event_indices
        pr_piano_seed = reconstruct_pr.place_event_level(pr_piano_seed, event_ind_reconstruction)
        quantization_write = quantization
    elif temporal_granularity == "event_level":
        quantization_write = 1
    write_path = generated_folder + '/piano_seed.mid'
    write_midi(pr_piano_seed, quantization_write, write_path, tempo=80)


if __name__ == '__main__':
    main_path = '/home/aciditeam-leo/Aciditeam/lop/Results/event_level/binary/quantization_100/rmsprop/Lstm'
    data_path = '../Data'
    track_paths = [
        '/home/aciditeam-leo/Aciditeam/database/Orchestration/Orchestration_checked/liszt_classical_archives/16',
        '/home/aciditeam-leo/Aciditeam/database/Orchestration/Orchestration_checked/bouliane/22',
        '/home/aciditeam-leo/Aciditeam/database/Orchestration/Orchestration_checked/bouliane/0',  # This one is in train set
    ]

    folders = glob.glob(main_path + '/[0-9]*')
    # for folder in folders:
    #     for track_path in track_paths:
    #         generate_keras(folder, data_path, track_path)

    for track_path in track_paths:
        generate_keras(main_path + '/5', data_path, track_path)
