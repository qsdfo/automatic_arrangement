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
from LOP.Utils.process_data import process_data_piano, process_data_orch
from LOP.Utils.normalization import apply_pca, apply_zca

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
    pr_piano, event_piano, duration_piano, _, name_piano, pr_orch, _, _, instru_orch, _, duration =\
        build_data_aux.process_folder(tracks_path, quantization, temporal_granularity)
    return pr_piano, event_piano, duration_piano, name_piano, pr_orch, instru_orch, duration

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


def generate_midi(config_folder, score_source, number_of_version, duration_gen, rhythmic_reconstruction, logger_generate):
    """This function generate the orchestration of a midi piano score
    
    Parameters
    ----------
    config_folder : str
        Absolute path to the configuration folder, i.e. the folder containing the saved model and the results
    score_source : str
        Either a path to a folder containing two midi files (piano and orchestration) or the path toa piano midi files
    number_of_version : int
        Number of version generated in a batch manner. Since the generation process involves sampling it might be interesting to generate several versions
    duration_gen : int
        Length of the generated score (in number of events). Useful for generating only the beginning of the piece.
    rhythmic_reconstruction: bool
        Whether rythmic reconstrcution from event-level representation to frame-level reconstrcution is performed or not. If true is selected, the rhtyhmic structure of the original piano score is used.
    logger_generate : logger
        Instanciation of logging. Can be None
    
    """

    logger_generate.info("#############################################")
    logger_generate.info("Orchestrating piano score : " + score_source)
    ############################################################
    # Load model, config and data
    ############################################################

    ########################
    # Load config and model
    parameters = pkl.load(open(config_folder + '/../script_parameters.pkl', 'rb'))
    model_parameters = pkl.load(open(config_folder + '/../model_params.pkl', 'rb'))
    # Set a minimum seed size, because for very short models you don't event see the beginning
    seed_size = max(model_parameters['temporal_order'], 10) - 1
    quantization = parameters['quantization']
    temporal_granularity = parameters['temporal_granularity']
    instru_mapping = parameters['instru_mapping']
    ########################

    ########################
    # Load data
    if re.search(ur'mid$', score_source):
        pr_piano, event_piano, duration_piano, name_piano, pr_orch, instru_orch, duration = load_solo(score_source, quantization, temporal_granularity)
    else:
        pr_piano, event_piano, duration_piano, name_piano, pr_orch, instru_orch, duration = load_from_pair(score_source, quantization, temporal_granularity)
    ########################

    ########################
    # Shorten
    # Keep only the beginning of the pieces (let's say a 100 events)
    pr_piano = extract_pianoroll_part(pr_piano, 0, duration_gen)
    duration_piano = duration_piano[:duration_gen]
    event_piano = event_piano[:duration_gen] 
    pr_orch = extract_pianoroll_part(pr_orch, 0, duration_gen)
    ########################

    ########################
    # Instanciate piano pianoroll
    N_piano = instru_mapping['Piano']['index_max']
    pr_piano_gen = np.zeros((duration_gen, N_piano), dtype=np.float32)
    pr_piano_gen = build_data_aux.cast_small_pr_into_big_pr(pr_piano, {}, 0, duration_gen, instru_mapping, pr_piano_gen)
    pr_piano_gen_flat = pr_piano_gen.sum(axis=1)
    silence_piano = [e for e in range(duration_gen) if pr_piano_gen_flat[e]== 0]
    ########################

    ########################
    # Instanciate orchestra pianoroll with orchestra seed
    N_orchestra = parameters['N_orchestra']
    if pr_orch:
        pr_orchestra_gen = np.zeros((seed_size, N_orchestra), dtype=np.float32)
        orch_seed_beginning = {k: v[:seed_size] for k, v in pr_orch.iteritems()}
        pr_orchestra_gen = build_data_aux.cast_small_pr_into_big_pr(orch_seed_beginning, instru_orch, 0, seed_size, instru_mapping, pr_orchestra_gen)
        pr_orchestra_truth = np.zeros((duration_gen, N_orchestra), dtype=np.float32)
        pr_orchestra_truth = build_data_aux.cast_small_pr_into_big_pr(pr_orch, instru_orch, 0, duration_gen, instru_mapping, pr_orchestra_truth)
    else:
        pr_orchestra_gen = None
        pr_orchestra_truth = None
    ########################
    
    ########################
    # Process data
    if duration_gen is None:
        duration_gen = duration
    pr_piano_gen = process_data_piano(pr_piano_gen, duration_piano, parameters)
    pr_orchestra_truth = process_data_orch(pr_orchestra_truth, parameters)
    pr_orchestra_gen = process_data_orch(pr_orchestra_gen, parameters)
    ########################
    
    ########################
    # Inputs' normalization
    if parameters["normalize"] is not None:
        if parameters["normalize"] == "standard_pca":
            # load whitening params
            standard_pca_piano = np.load(os.path.join(config_folder, "standard_pca_piano"))
            pr_piano_gen_norm = apply_pca(pr_piano_gen, standard_pca_piano['mean_piano'], standard_pca_piano['std_piano'], standard_pca_piano['pca_piano'], standard_pca_piano['epsilon']) 
        elif parameters["normalize"] == "standard_zca":
            standard_zca_piano = np.load(os.path.join(config_folder, "standard_zca_piano"))
            pr_piano_gen_norm = apply_zca(pr_piano_gen, standard_zca_piano['mean_piano'], standard_zca_piano['std_piano'], standard_zca_piano['zca_piano'], standard_zca_piano['epsilon'])
        else:
            raise Exception(str(parameters["normalize"]) + " is not a possible value for normalization parameter")        
    ########################
    
    ########################
    # Store folder
    string = re.split(r'/', name_piano)[-1]
    name_track = re.sub('piano_solo.mid', '', string)
    generated_folder = config_folder + '/generation_reference_example/' + name_track
    if not os.path.isdir(generated_folder):
        os.makedirs(generated_folder)
    ########################

    ############################################################
    # Generate
    ############################################################
    time_generate_0 = time.time()
    generated_sequences_Xent = generate(pr_piano_gen_norm, silence_piano, config_folder, 'model_Xent', pr_orchestra_gen, batch_size=number_of_version)
    generated_sequences_acc = generate(pr_piano_gen_norm, silence_piano, config_folder, 'model_acc', pr_orchestra_gen, batch_size=number_of_version)
    time_generate_1 = time.time()
    logger_generate.info('TTT : Generating data took {} seconds'.format(time_generate_1-time_generate_0))

    ############################################################
    # Reconstruct and write
    ############################################################
    def reconstruct_write_aux(generated_sequences, prefix):
        for write_counter in xrange(generated_sequences.shape[0]):
            # To distinguish when seed stop, insert a sustained note
            this_seq = generated_sequences[write_counter] * 127
            this_seq[:seed_size, 0] = 20
            # Reconstruct
            if rhythmic_reconstruction:
                pr_orchestra_clean = from_event_to_frame(this_seq, event_piano)
            else:
                pr_orchestra_clean = this_seq
            pr_orchestra = instrument_reconstruction(pr_orchestra_clean, instru_mapping)
            # Write
            write_path = generated_folder + '/' + prefix + '_' + str(write_counter) + '_generated.mid'
            if rhythmic_reconstruction:
                write_midi(pr_orchestra, quantization, write_path, tempo=80)
            else:
                write_midi(pr_orchestra, 1, write_path, tempo=80)
        return
    reconstruct_write_aux(generated_sequences_Xent, 'Xent')
    reconstruct_write_aux(generated_sequences_acc, 'acc')
    
    ############################################################
    ############################################################
    # Write original orchestration and piano scores, but reconstructed version, just to check
    if rhythmic_reconstruction:
        A = from_event_to_frame(pr_piano_gen, event_piano)
    else:
        A = pr_piano_gen
    B = A * 127
    if parameters['duration_piano']:
        B = B[:, :-1]  # Remove duration column
    piano_reconstructed = instrument_reconstruction_piano(B, instru_mapping)
    write_path = generated_folder + '/piano_reconstructed.mid'
    if rhythmic_reconstruction:
        write_midi(piano_reconstructed, quantization, write_path, tempo=80)
    else:
        write_midi(piano_reconstructed, 1, write_path, tempo=80)
    #
    if rhythmic_reconstruction:
        A = from_event_to_frame(pr_orchestra_truth, event_piano)
    else:
        A = pr_orchestra_truth
    B = A * 127
    orchestra_reconstructed = instrument_reconstruction(B, instru_mapping)
    write_path = generated_folder + '/orchestra_reconstructed.mid'
    if rhythmic_reconstruction:
        write_midi(orchestra_reconstructed, quantization, write_path, tempo=80)
    else:
        write_midi(orchestra_reconstructed, 1, write_path, tempo=80)
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