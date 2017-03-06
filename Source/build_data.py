#!/usr/bin/env python
# -*- coding: utf8 -*-


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
import numpy as np

from acidano.data_processing.utils.build_dico import build_dico
from acidano.data_processing.utils.pianoroll_processing import sum_along_instru_dim
from acidano.data_processing.utils.event_level import get_event_ind_dict
from acidano.data_processing.utils.time_warping import warp_pr_aux
import build_data_aux
import cPickle as pickle

from acidano.visualization.numpy_array.write_numpy_array_html import write_numpy_array_html
from acidano.visualization.numpy_array.dumped_numpy_to_csv import dump_to_csv


def aux(var, name, csv_path, html_path):
    np.savetxt(csv_path, var, delimiter=',')
    dump_to_csv(csv_path, csv_path)
    write_numpy_array_html(html_path, name)
    return


def get_dim_matrix(root_dir, index_files_dict, meta_info_path='temp.p', quantization=12, unit_type='binary', temporal_granularity='frame_level', logging=None):
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

        logging.info("##########")
        logging.info(set_identifier)
        # Get the full size of the tracks and instrument present
        T = 0
        for index_file in index_files:
            # Read the csv file indexing the database
            with open(index_file, 'rb') as f:
                for folder_path_relative in f:
                    folder_path = root_dir + '/' + folder_path_relative.rstrip()
                    logging.info(folder_path)
                    if not os.path.isdir(folder_path):
                        continue

                    # Read pr
                    pr_piano, instru_piano, name_piano, pr_orchestra, instru_orchestra, name_orchestra, duration =\
                        build_data_aux.process_folder(folder_path, quantization, unit_type, temporal_granularity, logging, gapopen=3, gapextend=1)

                    if duration is None:
                        # Files that could not be aligned
                        continue
                    T += duration

                    # Modify the mapping from instrument to indices in pianorolls and pitch bounds
                    instru_mapping = build_data_aux.instru_pitch_range(instrumentation=instru_piano,
                                                                       pr=pr_piano,
                                                                       instru_mapping=instru_mapping,
                                                                       instrument_list_from_dico=instrument_list_from_dico,
                                                                       )
                    # remark : instru_mapping would be modified if it is only passed to the function,
                    #                   f(a)  where a is modified inside the function
                    # but i prefer to make the reallocation explicit
                    #                   a = f(a) with f returning the modified value of a.
                    # Does it change anything for computation speed ? (Python pass by reference,
                    # but a slightly different version of it, not clear to me)
                    instru_mapping = build_data_aux.instru_pitch_range(instrumentation=instru_orchestra,
                                                                       pr=pr_orchestra,
                                                                       instru_mapping=instru_mapping,
                                                                       instrument_list_from_dico=instrument_list_from_dico
                                                                       )

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
    return instru_mapping, quantization, T_dict, counter


def cast_pr(new_pr_orchestra, new_instru_orchestra, new_pr_piano, start_time, duration, instru_mapping, pr_orchestra, pr_piano, logging=None):
    pr_orchestra = build_data_aux.cast_small_pr_into_big_pr(new_pr_orchestra, new_instru_orchestra, start_time, duration, instru_mapping, pr_orchestra)
    pr_piano = build_data_aux.cast_small_pr_into_big_pr(new_pr_piano, {}, start_time, duration, instru_mapping, pr_piano)


def build_data(root_dir, index_files_dict, meta_info_path='temp.p',quantization=12, unit_type='binary', temporal_granularity='frame_level', store_folder='../Data', logging=None):
    # Get dimensions
    instru_mapping, quantization, T_dict, N_orchestra = get_dim_matrix(root_dir, index_files_dict, meta_info_path=meta_info_path, quantization=quantization, unit_type=unit_type, temporal_granularity=temporal_granularity, logging=logging)

    statistics = {}

    temp = pickle.load(open(meta_info_path, 'rb'))
    instru_mapping = temp['instru_mapping']
    quantization = temp['quantization']
    T_dict = temp['T']
    N_orchestra = temp['N_orchestra']
    N_piano = instru_mapping['piano']['index_max']

    for set_identifier, index_files in index_files_dict.iteritems():
        T = T_dict[set_identifier]
        logging.info('T = ' + str(T))
        logging.info('N_orchestra = ' + str(N_orchestra))
        logging.info('N_piano = ' + str(N_piano))
        ########################################
        ########################################
        ########################################
        pr_orchestra = np.zeros((T, N_orchestra), dtype=np.float32)
        pr_piano = np.zeros((T, N_piano), dtype=np.float32)

        # Write the prs in the matrix
        time = 0
        tracks_start_end = {}

        # Browse index files and read pianoroll into the huge matrix
        for index_file in index_files:
            # Read the csv file indexing the database
            with open(index_file, 'rb') as f:
                for folder_path_relative in f:
                    folder_path = root_dir + '/' + folder_path_relative.rstrip()
                    logging.info(folder_path)
                    if not os.path.isdir(folder_path):
                        continue

                    # Get pr, warped and duration
                    new_pr_piano, new_instru_piano, name_piano, new_pr_orchestra, new_instru_orchestra, name_orchestra, duration\
                        = build_data_aux.process_folder(folder_path, quantization, unit_type, temporal_granularity, logging, gapopen=3, gapextend=1)

                    # SKip shitty files
                    if new_pr_piano is None:
                        # It's definitely not a match...
                        # Check for the files : are they really a piano score and its orchestration ??
                        with(open('log_build_db.txt', 'a')) as f:
                            f.write(folder_path + '\n')
                        continue

                    # and cast them in the appropriate bigger structure
                    cast_pr(new_pr_orchestra, new_instru_orchestra, new_pr_piano, time,
                            duration, instru_mapping, pr_orchestra, pr_piano, logging)

                    # Store beginning and end of this track
                    tracks_start_end[folder_path] = (time, time+duration)

                    # Increment time counter
                    time += duration

                    # Compute statistics
                    for track_name, instrument_name in new_instru_orchestra.iteritems():
                        # Number of note played by this instru
                        n_note_played = (new_pr_orchestra[track_name] > 0).sum()
                        if instrument_name in statistics:
                            # Track appearance
                            statistics[instrument_name]['n_track_present'] = statistics[instrument_name]['n_track_present'] + 1
                            statistics[instrument_name]['n_note_played'] = statistics[instrument_name]['n_note_played'] + n_note_played
                        else:
                            statistics[instrument_name] = {}
                            statistics[instrument_name]['n_track_present'] = 1
                            statistics[instrument_name]['n_note_played'] = n_note_played

        with open(store_folder + '/orchestra_' + set_identifier + '.csv', 'wb') as outfile:
            np.save(outfile, pr_orchestra)
        with open(store_folder + '/piano_' + set_identifier + '.csv', 'wb') as outfile:
            np.save(outfile, pr_piano)
        pickle.dump(tracks_start_end, open(store_folder + '/tracks_start_end_' + set_identifier + '.pkl', 'wb'))

        ####################################################################
        ####################################################################
        ####################################################################
        # aux(var=pr_piano,
        #     name='piano_' + set_identifier + '_' + temporal_granularity,
        #     csv_path='DEBUG/piano_' + set_identifier + '_' + temporal_granularity + '.csv',
        #     html_path='DEBUG/piano_' + set_identifier + '_' + temporal_granularity + '.html')
        #
        # aux(var=pr_orchestra,
        #     name='orchestra_' + set_identifier + '_' + temporal_granularity,
        #     csv_path='DEBUG/orchestra_' + set_identifier + '_' + temporal_granularity +'.csv',
        #     html_path='DEBUG/orchestra_' + set_identifier + '_' + temporal_granularity + '.html')
        ####################################################################
        ####################################################################
        ####################################################################

    # Save pr_orchestra, pr_piano, instru_mapping
    metadata = {}
    metadata['quantization'] = quantization
    metadata['N_orchestra'] = N_orchestra
    metadata['instru_mapping'] = instru_mapping
    with open(store_folder + '/metadata.pkl', 'wb') as outfile:
        pickle.dump(metadata, outfile)

    # Write statistics in a csv
    header = "instrument_name;n_track_present;n_note_played"
    with open(store_folder + '/statistics.csv', 'wb') as csvfile:
        csvfile.write(header+'\n')
        for instru_name, dico_stat in statistics.iteritems():
            csvfile.write(instru_name + u';' +
                          str(statistics[instru_name]['n_track_present']) + u';' +
                          str(statistics[instru_name]['n_note_played']) + '\n')


if __name__ == '__main__':
    import logging
    DATABASE_PATH = '/home/aciditeam-leo/Aciditeam/database/Orchestration/Orchestration_checked'
    data_folder = 'DEBUG'
    index_files_dict = {}
    index_files_dict['train'] = [
        # DATABASE_PATH + "/debug_train.txt",
        DATABASE_PATH + "/bouliane_train.txt",
        DATABASE_PATH + "/hand_picked_Spotify_train.txt",
        DATABASE_PATH + "/liszt_classical_archives_train.txt"
    ]
    index_files_dict['valid'] = [
        # DATABASE_PATH + "/debug_valid.txt",
        DATABASE_PATH + "/bouliane_valid.txt",
        DATABASE_PATH + "/hand_picked_Spotify_valid.txt",
        DATABASE_PATH + "/liszt_classical_archives_valid.txt"
    ]
    index_files_dict['test'] = [
        # DATABASE_PATH + "/debug_test.txt",
        DATABASE_PATH + "/bouliane_test.txt",
        DATABASE_PATH + "/hand_picked_Spotify_test.txt",
        DATABASE_PATH + "/liszt_classical_archives_test.txt"
    ]

    build_data(root_dir=DATABASE_PATH,
               index_files_dict=index_files_dict,
               meta_info_path=data_folder + '/temp.p',
               quantization=4,
               temporal_granularity='frame_level',
               store_folder=data_folder,
               logging=logging)
