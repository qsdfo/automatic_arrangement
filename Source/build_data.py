#!/usr/bin/env python
# -*- coding: utf8 -*-


#################################################
#################################################
#################################################
# Note :
# - for instrument pitch limits, we count discarding instrument mixes
#       then, casting the pr, we remove for instrumental mix the pitch outliers
#################################################
#################################################
#################################################


import os
import numpy as np
from acidano.data_processing.utils.build_dico import build_dico
from acidano.data_processing.utils.time_warping import needleman_chord_wrapper, warp_dictionnary_trace, remove_zero_in_trace, warp_pr_aux
from acidano.data_processing.utils.pianoroll_processing import sum_along_instru_dim
from acidano.data_processing.utils.event_level import get_event_ind_dict
import build_data_aux
import cPickle as pickle
import theano

from acidano.visualization.numpy_array.write_numpy_array_html import write_numpy_array_html
from acidano.visualization.numpy_array.dumped_numpy_to_csv import dump_to_csv


def aux(var, name, csv_path, html_path):
    np.savetxt(csv_path, var, delimiter=',')
    dump_to_csv(csv_path, csv_path)
    write_numpy_array_html(html_path, name)
    return


def get_dim_matrix(index_files_dict, meta_info_path='temp.p', quantization=12, temporal_granularity='frame_level'):
    # Determine the temporal size of the matrices
    # If the two files have different sizes, we use the shortest (to limit the use of memory,
    # we better contract files instead of expanding them).
    # Get instrument names
    instrument_list_from_dico = build_dico().keys()
    instru_mapping = {}
    # instru_mapping = {'piano': {'pitch_min': 24, 'pitch_max':117, 'ind_min': 0, 'ind_max': 92},
    #                         'harp' ... }
    T_dict = {}      # indexed by set_identifier
    for set_identifier, index_files in index_files_dict.iteritems():

        print("##########")
        print(set_identifier)
        # Get the full size of the tracks and instrument present
        T = 0
        for index_file in index_files:
            # Read the csv file indexing the database
            with open(index_file, 'rb') as f:
                for folder_path in f:
                    folder_path = folder_path.rstrip()
                    print folder_path
                    if not os.path.isdir(folder_path):
                        continue

                    # Read pr
                    try:
                        pr0, instru0, T0, name0, pr1, instru1, T1, name1 = build_data_aux.get_instru_and_pr_from_folder_path(folder_path, quantization)
                    except:
                        with open('log', 'wb') as f:
                            f.write('Bad file' + folder_path + '\n')
                        continue

                    # Temporal granularity
                    if temporal_granularity == 'event_level':
                        new_event_0 = get_event_ind_dict(pr0)
                        pr0 = warp_pr_aux(pr0, new_event_0)
                        new_event_1 = get_event_ind_dict(pr1)
                        pr1 = warp_pr_aux(pr1, new_event_1)

                    # Get T
                    trace_0, trace_1, this_sum_score, this_nbId, this_nbDiffs = needleman_chord_wrapper(sum_along_instru_dim(pr0), sum_along_instru_dim(pr1))
                    trace_prod = [e1 * e2 for (e1,e2) in zip(trace_0, trace_1)]
                    T += sum(trace_prod)
                    # Modify the mapping from instrument to indices in pianorolls and pitch bounds
                    instru_mapping = build_data_aux.instru_pitch_range(instrumentation=instru0,
                                                                       pr=pr0,
                                                                       instru_mapping=instru_mapping,
                                                                       instrument_list_from_dico=instrument_list_from_dico,
                                                                       )
                    # remark : instru_mapping would be modified if it is only passed to the function,
                    #                   f(a)  where a is modified inside the function
                    # but i prefer to make the reallocation explicit
                    #                   a = f(a) with f returning the modified value of a.
                    # Does it change anything for computation speed ? (Python pass by reference,
                    # but a slightly different version of it, not clear to me)
                    instru_mapping = build_data_aux.instru_pitch_range(instrumentation=instru1,
                                                                       pr=pr1,
                                                                       instru_mapping=instru_mapping,
                                                                       instrument_list_from_dico=instrument_list_from_dico
                                                                       )
                    # Delete prs
                    del pr0, pr1, instru0, instru1
        T_dict[set_identifier] = T

    # Build the index_min and index_max in the instru_mapping dictionary
    counter = 0
    for k, v in instru_mapping.iteritems():
        if k == 'piano':
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
    ########################################
    ########################################
    ########################################
    temp = {}
    temp['instru_mapping'] = instru_mapping
    temp['quantization'] = quantization
    temp['T'] = T_dict
    temp['N_orchestra'] = counter
    pickle.dump(temp, open(meta_info_path, 'wb'))
    return


def process_folder(folder_path, quantization, temporal_granularity):
    # Get instrus and prs from a folder name name
    pr0, instru0, _, name0, pr1, instru1, _, name1 = build_data_aux.get_instru_and_pr_from_folder_path(folder_path, quantization)

    # Temporal granularity
    if temporal_granularity == 'event_level':
        pr0 = warp_pr_aux(pr0, get_event_ind_dict(pr0))
        pr1 = warp_pr_aux(pr1, get_event_ind_dict(pr1))

    # Get trace from needleman_wunsch algorithm
    # Traces are binary lists, 0 meaning a gap is inserted
    trace_0, trace_1, this_sum_score, this_nbId, this_nbDiffs = needleman_chord_wrapper(sum_along_instru_dim(pr0), sum_along_instru_dim(pr1))

    # Wrap dictionnaries according to the traces
    assert(len(trace_0) == len(trace_1)), "size mismatch"
    pr0_warp = warp_dictionnary_trace(pr0, trace_0)
    pr1_warp = warp_dictionnary_trace(pr1, trace_1)

    # Get pr warped and duration# In fact we just discard 0 in traces for both pr
    trace_prod = [e1 * e2 for (e1,e2) in zip(trace_0, trace_1)]

    duration = sum(trace_prod)
    if duration == 0:
        return [None]*7
    pr0_aligned = remove_zero_in_trace(pr0_warp, trace_prod)
    pr1_aligned = remove_zero_in_trace(pr1_warp, trace_prod)

    return pr0_aligned, instru0, name0, pr1_aligned, instru1, name1, duration


def cast_pr(pr0, instru0, pr1, instru1, start_time, duration, instru_mapping, pr_orchestra, pr_piano):
    # Find which pr is orchestra, which one is piano
    if len(set(instru0.keys())) > len(set(instru1.keys())):
        # Add the small pr to the general structure
        # pr0 is orchestra
        pr_orchestra = build_data_aux.cast_small_pr_into_big_pr(pr0, instru0, start_time, duration, instru_mapping, pr_orchestra)
        pr_piano = build_data_aux.cast_small_pr_into_big_pr(pr1, {}, start_time, duration, instru_mapping, pr_piano)
    elif len(set(instru0.keys())) < len(set(instru1.keys())):
        # pr1 is orchestra
        pr_piano = build_data_aux.cast_small_pr_into_big_pr(pr0, {}, start_time, duration, instru_mapping, pr_piano)
        pr_orchestra = build_data_aux.cast_small_pr_into_big_pr(pr1, instru1, start_time, duration, instru_mapping, pr_orchestra)
    else:
        print('The two midi files have the same number of instruments')


def build_data(index_files_dict, meta_info_path='temp.p',quantization=12, temporal_granularity='frame_level', store_folder='../Data'):
    # Get dimensions
    get_dim_matrix(index_files_dict, meta_info_path=meta_info_path, quantization=quantization, temporal_granularity=temporal_granularity)

    temp = pickle.load(open(meta_info_path, 'rb'))
    instru_mapping = temp['instru_mapping']
    quantization = temp['quantization']
    T_dict = temp['T']
    N_orchestra = temp['N_orchestra']
    N_piano = instru_mapping['piano']['index_max']

    for set_identifier, index_files in index_files_dict.iteritems():
        T = T_dict[set_identifier]
        print 'T = ' + str(T)
        print 'N_orchestra = ' + str(N_orchestra)
        print 'N_piano = ' + str(N_piano)

        ########################################
        ########################################
        ########################################
        pr_orchestra = np.zeros((T, N_orchestra), dtype=theano.config.floatX)
        pr_piano = np.zeros((T, N_piano), dtype=theano.config.floatX)

        # Write the prs in the matrix
        time = 0
        tracks_start_end = {}

        # Browse index files and read pianoroll into the huge matrix
        for index_file in index_files:
            # Read the csv file indexing the database
            with open(index_file, 'rb') as f:
                for folder_path in f:
                    folder_path = folder_path.rstrip()
                    print folder_path
                    if not os.path.isdir(folder_path):
                        continue

                    # Get pr warped and duration
                    pr0, instru0, name0, pr1, instru1, name1, duration = process_folder(folder_path, quantization, temporal_granularity)

                    # SKip shitty files
                    if pr0 is None:
                        # It's definitely not a match...
                        # Check for the files : are they really an piano score and its orchestration ??
                        with(open('log.txt', 'a')) as f:
                            f.write(folder_path + '\n')
                        continue

                    # Find which pr is orchestra, which one is piano
                    # and cast them in the appropriate bigger structure
                    cast_pr(pr0, instru0, pr1, instru1, time, duration, instru_mapping, pr_orchestra, pr_piano)

                    # Store beginning and end of this track
                    tracks_start_end[folder_path] = (time, time+duration)

                    # Increment time counter
                    time += duration

        with open(store_folder + '/orchestra_' + set_identifier + '.csv', 'wb') as outfile:
            np.save(outfile, pr_orchestra)
        with open(store_folder + '/piano_' + set_identifier + '.csv', 'wb') as outfile:
            np.save(outfile, pr_piano)
        pickle.dump(tracks_start_end, open(store_folder + '/tracks_start_end_' + set_identifier + '.pkl', 'wb'))

        ####################################################################
        ####################################################################
        ####################################################################
        aux(var=pr_piano,
            name='piano_' + set_identifier + '_' + temporal_granularity,
            csv_path='DEBUG/piano_' + set_identifier + '_' + temporal_granularity + '.csv',
            html_path='DEBUG/piano_' + set_identifier + '_' + temporal_granularity + '.html')

        aux(var=pr_orchestra,
            name='orchestra_' + set_identifier + '_' + temporal_granularity,
            csv_path='DEBUG/orchestra_' + set_identifier + '_' + temporal_granularity +'.csv',
            html_path='DEBUG/orchestra_' + set_identifier + '_' + temporal_granularity + '.html')
        ####################################################################
        ####################################################################
        ####################################################################

    # Save pr_orchestra, pr_piano, instru_mapping
    metadata = {}
    metadata['quantization'] = quantization
    metadata['instru_mapping'] = instru_mapping
    with open(store_folder + '/metadata.pkl', 'wb') as outfile:
        pickle.dump(metadata, outfile)

if __name__ == '__main__':
    # subfolder_names = ['test']
    PREFIX_INDEX_FOLDER = "../Data/Index/"
    index_files_dict = {}
    index_files_dict['train'] = [
        PREFIX_INDEX_FOLDER + "debug_train.txt",
    ]
    index_files_dict['valid'] = [
        PREFIX_INDEX_FOLDER + "debug_valid.txt",
    ]
    index_files_dict['test'] = [
        PREFIX_INDEX_FOLDER + "debug_test.txt",
    ]
    build_data(index_files_dict=index_files_dict, meta_info_path='temp.p', quantization=12, temporal_granularity='event_level')
