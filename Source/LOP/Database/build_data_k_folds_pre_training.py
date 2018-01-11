#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

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
import shutil
import numpy as np
import LOP.Scripts.config as config
import build_data_aux
import build_data_aux_no_piano
import cPickle as pickle
from avoid_tracks import avoid_tracks

# memory issues
import gc
# gc.collect()
from guppy import hpy; hp = hpy()
import sys

DEBUG = False

def update_instru_mapping(folder_path, instru_mapping, T, quantization, is_piano):
    logging.info(folder_path)
    if not os.path.isdir(folder_path):
        return instru_mapping, T
    
    # Read pr
    if is_piano:
        pr_piano, _, _, instru_piano, _, pr_orch, _, _, instru_orch, _, duration =\
            build_data_aux.process_folder(folder_path, quantization, temporal_granularity, gapopen=3, gapextend=1)
    else:
        pr_piano, _, _, instru_piano, _, pr_orch, _, _, instru_orch, _, duration =\
            build_data_aux_no_piano.process_folder_NP(folder_path, quantization, temporal_granularity)
    
    if duration is None:
        # Files that could not be aligned
        return instru_mapping, T
    
    T += duration
    
    # Modify the mapping from instrument to indices in pianorolls and pitch bounds
    instru_mapping = build_data_aux.instru_pitch_range(instrumentation=instru_piano,
                                                       pr=pr_piano,
                                                       instru_mapping=instru_mapping,
                                                       )
    # remark : instru_mapping would be modified if it is only passed to the function,
    #                   f(a)  where a is modified inside the function
    # but i prefer to make the reallocation explicit
    #                   a = f(a) with f returning the modified value of a.
    # Does it change anything for computation speed ? (Python pass by reference,
    # but a slightly different version of it, not clear to me)
    instru_mapping = build_data_aux.instru_pitch_range(instrumentation=instru_orch,
                                                       pr=pr_orch,
                                                       instru_mapping=instru_mapping,
                                                       )
    return instru_mapping, T


def get_dim_matrix(folder_paths, folder_paths_pretraining, meta_info_path='temp.pkl', quantization=12, temporal_granularity='frame_level', T_limit=1e10, logging=None):
    logging.info("##########")
    logging.info("Get dimension informations")
    # Determine the temporal size of the matrices
    # If the two files have different sizes, we use the shortest (to limit the use of memory,
    # we better contract files instead of expanding them).
    # Get instrument names
    instru_mapping = {}
    # instru_mapping = {'piano': {'pitch_min': 24, 'pitch_max':117, 'ind_min': 0, 'ind_max': 92},
    #                         'harp' ... }
    T_splits = []
    T_pretraining_splits = []
    folder_paths_splits = []
    T_pretraining_splits = []
    folder_paths_pretraining_splits = []

    T = 0
    folder_paths_split = []
    T_pretraining = 0
    folder_paths_pretraining_split = []
    for folder_path in folder_paths:
        if T>T_limit:
            T_splits.append(T)
            folder_paths_splits.append(folder_paths_split)
            T = 0
            folder_paths_split = []
        folder_path = folder_path.rstrip()
        instru_mapping, T = update_instru_mapping(folder_path, instru_mapping, T, quantization, is_piano=True)
        folder_paths_split.append(folder_path)
    # Don't forget the last one !
    T_splits.append(T)
    folder_paths_splits.append(folder_paths_split)
        
    for folder_path_pre in folder_paths_pretraining:
        if T_pretraining>T_limit:
            T_pretraining_splits.append(T_pretraining)
            folder_paths_pretraining_splits.append(folder_paths_pretraining_split)
            T_pretraining = 0
            folder_paths_pretraining_split = []
        folder_path_pre = folder_path_pre.rstrip()
        instru_mapping, T_pretraining = update_instru_mapping(folder_path_pre, instru_mapping, T_pretraining, quantization, is_piano=False)
        folder_paths_pretraining_split.append(folder_path_pre)
    if len(folder_paths_pretraining) > 0:
        # Don't forget the last one !
        T_pretraining_splits.append(T_pretraining)
        folder_paths_pretraining_splits.append(folder_paths_pretraining_split)
        
    # Build the index_min and index_max in the instru_mapping dictionary
    counter = 0
    for k, v in instru_mapping.iteritems():
        if k == 'Piano':
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
    temp = {}
    temp['instru_mapping'] = instru_mapping
    temp['quantization'] = quantization
    temp['T_splits'] = T_splits
    temp['folder_paths_splits'] = folder_paths_splits
    temp['T_pretraining_splits'] = T_pretraining_splits
    temp['folder_paths_pretraining_splits'] = folder_paths_pretraining_splits
    temp['N_orchestra'] = counter
    pickle.dump(temp, open(meta_info_path, 'wb'))
    return


def cast_pr(new_pr_orchestra, new_instru_orchestra, new_pr_piano, start_time, duration, instru_mapping, 
            pr_orchestra, pr_piano, mask_orch, logging=None):
    pr_orchestra = build_data_aux.cast_small_pr_into_big_pr(new_pr_orchestra, new_instru_orchestra, start_time, duration, instru_mapping, pr_orchestra)
    pr_piano = build_data_aux.cast_small_pr_into_big_pr(new_pr_piano, {}, start_time, duration, instru_mapping, pr_piano)
    
    list_instru = new_instru_orchestra.values()
    for instru in list_instru:
        if instru == 'Remove':
            continue
        instru_names = build_data_aux.unmixed_instru(instru)    
        for instru_name in instru_names:
            ind_bot = instru_mapping[instru_name]['index_min']
            ind_top = instru_mapping[instru_name]['index_max']
            mask_orch[start_time:start_time+duration, ind_bot:ind_top] = 1
    return

def build_training_matrix(folder_paths, instru_mapping, 
                          pr_piano, pr_orchestra,
                          duration_piano, duration_orch,
                          mask_orch,
                          statistics,
                          is_piano): 
    # Write the prs in the matrix
    time = 0
    tracks_start_end = {}

    for folder_path in folder_paths:
        folder_path = folder_path.rstrip()
        logging.info(folder_path)
        if not os.path.isdir(folder_path):
            continue

        # Get pr, warped and duration
        if is_piano:
            new_pr_piano, _, new_duration_piano, _, _, new_pr_orchestra, _, new_duration_orch, new_instru_orchestra, _, duration\
                = build_data_aux.process_folder(folder_path, quantization, temporal_granularity, gapopen=3, gapextend=1)
        else:
            new_pr_piano, _, new_duration_piano, _, _, new_pr_orchestra, _, new_duration_orch, new_instru_orchestra, _, duration\
                = build_data_aux_no_piano.process_folder_NP(folder_path, quantization, temporal_granularity)

        # SKip shitty files
        if new_pr_piano is None:
            # It's definitely not a match...
            # Check for the files : are they really a piano score and its orchestration ??
            with(open('log_build_db.txt', 'a')) as f:
                f.write(folder_path + '\n')
            continue
        
        # and cast them in the appropriate bigger structure
        cast_pr(new_pr_orchestra, new_instru_orchestra, new_pr_piano, time,
                duration, instru_mapping, pr_orchestra, pr_piano, mask_orch, logging)

        duration_piano[time:time+duration] = new_duration_piano
        duration_orch[time:time+duration] = new_duration_orch

        # Store beginning and end of this track
        tracks_start_end[folder_path] = (time, time+duration)

        # Increment time counter
        time += duration

        # Compute statistics
        for track_name, instrument_name in new_instru_orchestra.iteritems():
            # Number of note played by this instru
            if track_name not in new_pr_orchestra.keys():
                continue
            n_note_played = (new_pr_orchestra[track_name] > 0).sum()
            if instrument_name in statistics:
                # Track appearance
                statistics[instrument_name]['n_track_present'] = statistics[instrument_name]['n_track_present'] + 1
                statistics[instrument_name]['n_note_played'] = statistics[instrument_name]['n_note_played'] + n_note_played
            else:
                statistics[instrument_name] = {}
                statistics[instrument_name]['n_track_present'] = 1
                statistics[instrument_name]['n_note_played'] = n_note_played

        del(new_pr_piano)
        del(new_pr_orchestra)
        del(new_instru_orchestra)
        gc.collect()

    return pr_piano, pr_orchestra, duration_piano, duration_orch, statistics, tracks_start_end

def build_data(folder_paths, folder_paths_pretraining, meta_info_path='temp.pkl', quantization=12, temporal_granularity='frame_level', store_folder='../Data', pitch_translation_augmentations=[0], logging=None):

    # Get dimensions
    if DEBUG:
        T_limit = 5000
    else:
        T_limit = 1e6
    
    get_dim_matrix(folder_paths, folder_paths_pretraining, meta_info_path=meta_info_path, quantization=quantization, temporal_granularity=temporal_granularity, T_limit=T_limit, logging=logging)

    logging.info("##########")
    logging.info("Build data")

    statistics = {}
    statistics_pretraining = {}

    temp = pickle.load(open(meta_info_path, 'rb'))
    instru_mapping = temp['instru_mapping']
    quantization = temp['quantization']
    T_splits = temp['T_splits']
    folder_paths_splits = temp['folder_paths_splits']
    T_pretraining_splits = temp['T_pretraining_splits']
    folder_paths_pretraining_splits = temp['folder_paths_pretraining_splits']
    N_orchestra = temp['N_orchestra']
    N_piano = instru_mapping['Piano']['index_max']

    for counter_split, (T_split, folder_paths_split) in enumerate(zip(T_splits, folder_paths_splits)):
        pr_orchestra = np.zeros((T_split, N_orchestra), dtype=np.float32)
        pr_piano = np.zeros((T_split, N_piano), dtype=np.float32)
        duration_piano = np.zeros((T_split), dtype=np.int)
        duration_orch = np.zeros((T_split), dtype=np.int)
        mask_orch = np.zeros((T_split, N_orchestra), dtype=np.int)

        pr_piano, pr_orchestra, duration_piano, duration_orch, \
        statistics, tracks_start_end = \
            build_training_matrix(folder_paths_split, instru_mapping, 
                              pr_piano, pr_orchestra,
                              duration_piano, duration_orch,
                              mask_orch,
                              statistics,
                              is_piano=True)

        with open(store_folder + '/orchestra_' + str(counter_split) + '.npy', 'wb') as outfile:
            np.save(outfile, pr_orchestra)
        with open(store_folder + '/piano_' + str(counter_split) + '.npy', 'wb') as outfile:
            np.save(outfile, pr_piano)
        with open(store_folder + '/duration_orchestra_' + str(counter_split) + '.npy', 'wb') as outfile:
            np.save(outfile, duration_orch)
        with open(store_folder + '/duration_piano_' + str(counter_split) + '.npy', 'wb') as outfile:
            np.save(outfile, duration_piano)
        with open(store_folder + '/mask_orch_' + str(counter_split) + '.npy', 'wb') as outfile:
            np.save(outfile, mask_orch)
        pickle.dump(tracks_start_end, open(store_folder + '/tracks_start_end_' + str(counter_split) + '.pkl', 'wb'))

        # print(str(counter_split) + " : ")
        # print(hp.heap())
        # print("Number of objects tracked by GC : " + str(len(gc.get_objects())))

    # Explicitely free memory
    del(pr_piano)
    del(pr_orchestra) 
    del(duration_piano)
    del(duration_orch)
    del(tracks_start_end)
    gc.collect()

    for counter_split, (T_pretraining_split, folder_paths_pretraining_split) in enumerate(zip(T_pretraining_splits, folder_paths_pretraining_splits)):
        pr_orchestra_pretraining = np.zeros((T_pretraining_split, N_orchestra), dtype=np.float32)
        pr_piano_pretraining = np.zeros((T_pretraining_split, N_piano), dtype=np.float32)
        duration_piano_pretraining = np.zeros((T_pretraining_split), dtype=np.int)
        duration_orch_pretraining = np.zeros((T_pretraining_split), dtype=np.int)
        mask_orch_pretraining = np.zeros((T_pretraining_split, N_orchestra), dtype=np.int)
        
        pr_piano_pretraining, pr_orchestra_pretraining, duration_piano_pretraining, duration_orch_pretraining, \
        statistics_pretraining, tracks_start_end_pretraining = \
            build_training_matrix(folder_paths_pretraining_split, instru_mapping, 
                              pr_piano_pretraining, pr_orchestra_pretraining,
                              duration_piano_pretraining, duration_orch_pretraining,
                              mask_orch_pretraining,
                              statistics_pretraining,
                              is_piano=False)
    
        with open(store_folder + '/orchestra_pretraining_' + str(counter_split) + '.npy', 'wb') as outfile:
            np.save(outfile, pr_orchestra_pretraining)
        with open(store_folder + '/piano_pretraining_' + str(counter_split) + '.npy', 'wb') as outfile:
            np.save(outfile, pr_piano_pretraining)
        with open(store_folder + '/duration_orchestra_pretraining_' + str(counter_split) + '.npy', 'wb') as outfile:
            np.save(outfile, duration_orch_pretraining)
        with open(store_folder + '/duration_piano_pretraining_' + str(counter_split) + '.npy', 'wb') as outfile:
            np.save(outfile, duration_piano_pretraining)
        with open(store_folder + '/mask_orch_pretraining_' + str(counter_split) + '.npy', 'wb') as outfile:
            np.save(outfile, mask_orch_pretraining)
        pickle.dump(tracks_start_end_pretraining, open(store_folder + '/tracks_start_end_pretraining_' + str(counter_split) + '.pkl', 'wb'))

    del(pr_piano_pretraining)
    del(pr_orchestra_pretraining) 
    del(duration_piano_pretraining)
    del(duration_orch_pretraining)
    del(tracks_start_end_pretraining)
    gc.collect()

    # Save pr_orchestra, pr_piano, instru_mapping
    metadata = {}
    metadata['quantization'] = quantization
    metadata['N_orchestra'] = N_orchestra
    metadata['N_piano'] = N_piano
    metadata['instru_mapping'] = instru_mapping
    metadata['quantization'] = quantization
    metadata['temporal_granularity'] = temporal_granularity
    metadata['store_folder'] = store_folder

    with open(store_folder + '/metadata.pkl', 'wb') as outfile:
        pickle.dump(metadata, outfile)

    # Write statistics in a csv
    def write_statistics(namefile, data):
        header = "instrument_name;n_track_present;n_note_played"
        with open(store_folder + '/' + namefile + '.csv', 'wb') as csvfile:
            csvfile.write(header + '\n')
            for instru_name, dico_stat in data.iteritems():
                csvfile.write(instru_name + u';' +
                              str(dico_stat['n_track_present']) + u';' +
                              str(dico_stat['n_note_played']) + '\n')
        return
    write_statistics('statistics', statistics)
    write_statistics('statistics_pretraining', statistics_pretraining)

if __name__ == '__main__':
    import logging
    # log file
    log_file_path = 'log_build_data'
    # set up logging to file - see previous section for more details
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

    # Set up
    # NOTE : can't do data augmentation with K-folds, or it would require to build K times the database
    # because train is data augmented but not test and validate
    temporal_granularity = 'event_level'
    quantization = 8
    pretraining_bool = True

    # Database have to be built jointly so that the ranges match
    DATABASE_PATH = os.path.join(config.database_root(), 'LOP_database_06_09_17')
    if DEBUG:
        DATABASE_NAMES = ["debug"] #, "imslp"]
    else:
        DATABASE_NAMES = ["bouliane", "hand_picked_Spotify", "liszt_classical_archives"] # , "imslp"]
    DATABASE_PATH_PRETRAINING = os.path.join(config.database_pretraining_root(), 'SOD')
    if DEBUG:
        DATABASE_NAMES_PRETRAINING = ["debug"]
    else:
        DATABASE_NAMES_PRETRAINING = ["Kunstderfuge", "Musicalion", "Mutopia", "OpenMusicScores"]

    data_folder = '../../../Data/Data'
    if DEBUG:
        data_folder += '_DEBUG'
    if pretraining_bool:
        data_folder += '_pretraining'
    data_folder += '_tempGran' + str(quantization)
    
    if os.path.isdir(data_folder):
        shutil.rmtree(data_folder)    
    os.makedirs(data_folder)

    # Create a list of paths
    def build_filepaths_list(db_path=DATABASE_PATH, db_names=DATABASE_NAMES):
        folder_paths = []
        for db_name in db_names:
            path = db_path + '/' + db_name
            for file_name in os.listdir(path):
                if file_name != '.DS_Store':
                    this_path = db_name + '/' + file_name
                    folder_paths.append(this_path)
        return folder_paths
    
    folder_paths = build_filepaths_list(DATABASE_PATH, DATABASE_NAMES)
    folder_paths = [os.path.join(DATABASE_PATH, e) for e in folder_paths]
    
    if pretraining_bool:
        folder_paths_pretraining = build_filepaths_list(DATABASE_PATH_PRETRAINING, DATABASE_NAMES_PRETRAINING)
        folder_paths_pretraining = [os.path.join(DATABASE_PATH_PRETRAINING, e) for e in folder_paths_pretraining]
    else:
        folder_paths_pretraining = []

    # Remove garbage tracks
    avoid_tracks_list = avoid_tracks()
    folder_paths = [e for e in folder_paths if e not in avoid_tracks_list]
    folder_paths_pretraining = [e for e in folder_paths_pretraining if e not in avoid_tracks_list]

    build_data(folder_paths=folder_paths,
               folder_paths_pretraining=folder_paths_pretraining,
               meta_info_path=data_folder + '/temp.pkl',
               quantization=quantization,
               temporal_granularity=temporal_granularity,
               store_folder=data_folder,
               logging=logging)
