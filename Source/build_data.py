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
from acidano.data_processing.utils.pianoroll_processing import warp_pr
import build_data_aux
import cPickle as pickle


def get_dim_matrix(path_db, subfolder_names, quantization, meta_info_path='temp.p'):
    # Determine the temporal size of the matrices
    # If the two files have different sizes, we use the shortest (to limit the use of memory,
    # we better contract files instead of expanding them).
    # Get instrument names
    instrument_list_from_dico = build_dico().keys()
    instrument_mapping = {}
    # instrument_mapping = {'piano': {'pitch_min': 24, 'pitch_max':117, 'ind_min': 0, 'ind_max': 92},
    #                         'harp' ... }
    # Get the full size of the tracks and instrument present
    T = 0
    for sub_db in subfolder_names:
        print '#' * 30
        print sub_db
        sub_db_path = path_db + '/' + sub_db
        if not os.path.isdir(sub_db_path):
            continue
        for folder_name in os.listdir(sub_db_path):
            print '#' * 20
            print '#' + folder_name + '\n'
            folder_path = sub_db_path + '/' + folder_name
            if not os.path.isdir(folder_path):
                continue
            # Get instrus and prs from a folder name
            try:
                pr0, instru0, T0, pr1, instru1, T1 = build_data_aux.get_instru_and_pr_from_folder_path(folder_path, quantization)
            except:
                with open('log', 'wb') as f:
                    f.write('Bad file' + folder_path + '\n')
                continue
            T += min(T0, T1)
            # Modify the mapping from instrument to indices in pianorolls and pitch bounds
            instrument_mapping = build_data_aux.instru_pitch_range(instrumentation=instru0,
                                                                   pr=pr0,
                                                                   instrument_mapping=instrument_mapping,
                                                                   instrument_list_from_dico=instrument_list_from_dico,
                                                                   )
            # remark : instrument_mapping would be modified if it is only passed to the function,
            #                   f(a)  where a is modified inside the function
            # but i prefer to make the reallocation explicit
            #                   a = f(a) with f returning the modified value of a.
            # Does it change anything for computation speed ?
            instrument_mapping = build_data_aux.instru_pitch_range(instrumentation=instru1,
                                                                   pr=pr1,
                                                                   instrument_mapping=instrument_mapping,
                                                                   instrument_list_from_dico=instrument_list_from_dico
                                                                   )
            # Delete prs
            del pr0, pr1, instru0, instru1

    # Build the index_min and index_max in the instrument_mapping dictionary
    counter = 0
    for k, v in instrument_mapping.iteritems():
        if k == 'piano':
            index_min = 0
            index_max = v['pitch_max'] - v['pitch_min']
            instrument_mapping[k]['index_min'] = index_min
            instrument_mapping[k]['index_max'] = index_max
            continue
        index_min = counter
        counter = counter + v['pitch_max'] - v['pitch_min']
        index_max = counter
        instrument_mapping[k]['index_min'] = index_min
        instrument_mapping[k]['index_max'] = index_max

    # Instanciate the matrices
    ########################################
    ########################################
    ########################################
    temp = {}
    temp['instrument_mapping'] = instrument_mapping
    temp['quantization'] = quantization
    temp['T'] = T
    temp['counter'] = counter
    pickle.dump(temp, open(meta_info_path, 'wb'))
    return


def build_data(path_db, subfolder_names, save_path='../Data/data_midi.p', meta_info_path='temp.p'):
    temp = pickle.load(open(meta_info_path, 'rb'))
    instrument_mapping = temp['instrument_mapping']
    quantization = temp['quantization']
    T = temp['T']
    counter = temp['counter']

    print 'T = ' + str(T)
    ########################################
    ########################################
    ########################################
    pr_orchestra = np.zeros((T, counter), dtype=np.int16)
    pr_piano = np.zeros((T, instrument_mapping['piano']['index_max']), dtype=np.int16)

    # Write the prs in the matrix
    time = 0
    change_track = []
    for sub_db in subfolder_names:
        print '#' * 30
        print sub_db
        sub_db_path = path_db + '/' + sub_db
        if not os.path.isdir(sub_db_path):
            continue
        for folder_name in os.listdir(sub_db_path):
            print '#' * 20
            print '#' + folder_name + '\n'
            folder_path = sub_db_path + '/' + folder_name
            if not os.path.isdir(folder_path):
                continue
            # It's a new track ! Let's write its time index in change_track !!
            change_track.append(time)
            # Get instrus and prs from a folder name name
            pr0, instru0, T0, pr1, instru1, T1 = build_data_aux.get_instru_and_pr_from_folder_path(folder_path, quantization)
            duration = min(T0, T1)
            # !! Time warping !! (tonnerre et tout)     :)
            # In fact just do a linear warping          :(
            if T0 > T1:
                pr0 = warp_pr(pr0, T_target=T1)
            elif T0 < T1:
                pr1 = warp_pr(pr1, T_target=T0)
            # Do nothing if T0 = T1

            # Find which pr is orchestra, which one is piano
            if len(set(instru0.keys())) > len(set(instru1.keys())):
                # Add the small pr to the general structure
                # pr0 is orchestra
                pr_orchestra = build_data_aux.cast_small_pr_into_big_pr(pr0, instru0, time, duration, instrument_mapping, pr_orchestra)
                pr_piano = build_data_aux.cast_small_pr_into_big_pr(pr1, {}, time, duration, instrument_mapping, pr_piano)
            elif len(set(instru0.keys())) < len(set(instru1.keys())):
                # pr1 is orchestra
                pr_piano = build_data_aux.cast_small_pr_into_big_pr(pr0, {}, time, duration, instrument_mapping, pr_piano)
                pr_orchestra = build_data_aux.cast_small_pr_into_big_pr(pr1, instru1, time, duration, instrument_mapping, pr_orchestra)
            else:
                print('The two midi files have the same number of instruments')
            time += duration
            del pr0, pr1, instru0, instru1

    # Save pr_orchestra, pr_piano, instrument_mapping
    metadata = {}
    metadata['quantization'] = quantization
    metadata['instru_mapping'] = instrument_mapping
    metadata['change_track'] = change_track
    with open('../Data/metadata.pkl', 'wb') as outfile:
        pickle.dump(metadata, outfile)
    with open('../Data/pr_orchestra.csv', 'wb') as outfile:
        np.save(outfile, pr_orchestra)
    with open('../Data/pr_piano.csv', 'wb') as outfile:
        np.save(outfile, pr_piano)

if __name__ == '__main__':
    folder_path = '/Users/leo/Recherche/GitHub_Aciditeam/database/Orchestration/Orchestration_checked'
    subfolder_names = [
        'liszt_classical_archives',
        'bouliane',
        'hand_picked_Spotify',
    ]
    # subfolder_names = ['test']
    # get_dim_matrix(folder_path, subfolder_names, quantization=12, meta_info_path='temp.p')
    print '#'*50
    print '#'*50
    print '#'*50
    build_data(folder_path, subfolder_names, save_path='../Data/data_midi.p', meta_info_path='temp.p')
