#!/usr/bin/env python
# -*- coding: utf-8-unix -*-


import logging
import cPickle as pkl
import re
import numpy as np
import os
import time

from generate import generate

import LOP.Database.build_data_aux as build_data_aux

from LOP_database.midi.read_midi import Read_midi
from LOP_database.midi.write_midi import write_midi
from LOP_database.utils.pianoroll_processing import get_pianoroll_time, extract_pianoroll_part
from LOP_database.utils.event_level import get_event_ind_dict, from_event_to_frame
from LOP_database.utils.time_warping import warp_pr_aux
from LOP_database.utils.reconstruct_pr import instrument_reconstruction, instrument_reconstruction_piano

def load_from_pair(tracks_path, quantization, temporal_granularity):
    ############################################################
    # Read piano midi file and orchestra score if defined
    ############################################################
    pr_piano, event_piano, _, name_piano, pr_orch, _, instru_orch, _, duration =\
        build_data_aux.process_folder(tracks_path, quantization, temporal_granularity)
    return pr_piano, event_piano, name_piano, pr_orch, instru_orch, duration

def load_solo(piano_midi, quantization, temporal_granularity):
    # Read piano pr
    pr_piano = Read_midi(path, quantization).read_file()
    # Take event level representation
    if temporal_granularity == 'event_level':
        event_piano = get_event_ind_dict(pr_piano)
        pr_piano = warp_pr_aux(pr_piano, event_piano)
    else:
        event_piano = None

    name_piano = re.sub(ur'/.*\.mid', '', piano_midi)

    duration = get_pianoroll_time(pr_piano)

    return pr_piano, event_piano, name_piano, None, None, duration


def generate_midi(config_folder, score_source, number_of_version, logger_generate):
    # This function generate the orchestration of a midi piano score

    logger_generate.info("#############################################")
    logger_generate.info("Orchestrating piano score : " + score_source)
    ############################################################
    # Load the model and config
    ############################################################
    parameters = pkl.load(open(config_folder + '/../script_parameters.pkl', 'rb'))
    model_parameters = pkl.load(open(config_folder + '/../model_params.pkl', 'rb'))
    # Set a minimum seed size, because for very short models you don't event see the beginning
    seed_size = max(model_parameters['temporal_order'], 10) - 1
    quantization = parameters['quantization']
    temporal_granularity = parameters['temporal_granularity']
    instru_mapping = parameters['instru_mapping']

    if re.search(ur'mid$', score_source):
        pr_piano, event_piano, name_piano, pr_orch, instru_orch, duration = load_solo(score_source, quantization, temporal_granularity)
    else:
        pr_piano, event_piano, name_piano, pr_orch, instru_orch, duration = load_from_pair(score_source, quantization, temporal_granularity)

    # Keep only the beginning of the pieces (let's say a 100 events)
    duration = 100
    pr_piano = extract_pianoroll_part(pr_piano, 0, 100)
    event_piano = event_piano[:100]
    pr_orch = extract_pianoroll_part(pr_orch, 0, 100)

    # Instanciate piano pianoroll
    N_piano = instru_mapping['Piano']['index_max']
    pr_piano_gen = np.zeros((duration, N_piano), dtype=np.float32)
    pr_piano_gen = build_data_aux.cast_small_pr_into_big_pr(pr_piano, {}, 0, duration, instru_mapping, pr_piano_gen)

    # Instanciate orchestra pianoroll with orchestra seed
    N_orchestra = parameters['N_orchestra']
    if pr_orch:
        pr_orchestra_gen = np.zeros((seed_size, N_orchestra), dtype=np.float32)
        orch_seed_beginning = {k: v[:seed_size] for k, v in pr_orch.iteritems()}
        pr_orchestra_gen = build_data_aux.cast_small_pr_into_big_pr(orch_seed_beginning, instru_orch, 0, seed_size, instru_mapping, pr_orchestra_gen)
        pr_orchestra_truth = np.zeros((duration, N_orchestra), dtype=np.float32)
        pr_orchestra_truth = build_data_aux.cast_small_pr_into_big_pr(pr_orch, instru_orch, 0, duration, instru_mapping, pr_orchestra_truth)
    else:
        pr_orchestra_gen = None
        pr_orchestra_truth = None

    # Were data binarized ?
    if parameters["binarize_piano"]:
        pr_piano_gen[np.nonzero(pr_piano_gen)] = 1
    if parameters["binarize_orchestra"]:
        if pr_orchestra_gen is not None:
            pr_orchestra_gen[np.nonzero(pr_orchestra_gen)] = 1
            pr_orchestra_truth[np.nonzero(pr_orchestra_truth)] = 1

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
    generated_sequences = generate(pr_piano_gen, config_folder, pr_orchestra_gen, batch_size=number_of_version)
    time_generate_1 = time.time()
    logger_generate.info('TTT : Generating data took {} seconds'.format(time_generate_1-time_generate_0))

    ############################################################
    # Reconstruct and write
    ############################################################
    for write_counter in xrange(generated_sequences.shape[0]):
        # To distinguish when seed stop, insert a sustained note
        this_seq = generated_sequences[write_counter] * 127
        this_seq[:seed_size, 0] = 1
        # Reconstruct
        pr_orchestra_clean = from_event_to_frame(this_seq, event_piano)
        pr_orchestra = instrument_reconstruction(pr_orchestra_clean, instru_mapping)
        # Write
        write_path = generated_folder + '/' + str(write_counter) + '_generated.mid'
        write_midi(pr_orchestra, quantization, write_path, tempo=80)

    ############################################################
    ############################################################
    # Write original orchestration and piano scores, but reconstructed version, just to check
    A = from_event_to_frame(pr_piano_gen, event_piano)
    B = A * 127
    piano_reconstructed = instrument_reconstruction_piano(B, instru_mapping)
    write_path = generated_folder + '/piano_reconstructed.mid'
    write_midi(piano_reconstructed, quantization, write_path, tempo=80)
    #
    A = from_event_to_frame(pr_orchestra_truth, event_piano)
    B = A * 127
    orchestra_reconstructed = instrument_reconstruction(B, instru_mapping)
    write_path = generated_folder + '/orchestra_reconstructed.mid'
    write_midi(orchestra_reconstructed, quantization, write_path, tempo=80)
    ############################################################
    ############################################################

if __name__ == '__main__':
    config_folder = '/Users/leo/Recherche/GitHub_Aciditeam/lop/Results/Data__event_level8/LSTM_plugged_base/0/0'
    score_sources = ['/Users/leo/Recherche/GitHub_Aciditeam/database/Orchestration/LOP_database_06_09_17/liszt_classical_archives/16',
        # '/home/mil/leo/Database/LOP_database_06_09_17/liszt_classical_archives/17',
        # '/home/mil/leo/Database/LOP_database_06_09_17/liszt_classical_archives/18'
        ]

    for score_source in score_sources:
        generate_midi(config_folder, score_source, 2, logging)