#!/usr/bin/env python
# -*- coding: utf8 -*-


#################################################
#################################################
#################################################
# TODO:
# change_track
#################################################
#################################################
#################################################


import os
import numpy as np
from acidano.data_processing.utils.build_dico import build_dico
import build_data_aux
import cPickle as pickle


def build_data(path_db, subfolder_names, quantization, granularity='frame_level', save_path='../Data/data_midi.p'):
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
        print sub_db
        sub_db_path = path_db + '/' + sub_db
        if not os.path.isdir(sub_db_path):
            continue
        for folder_name in os.listdir(sub_db_path):
            print '#' * 30
            print '#' + folder_name + '\n'
            folder_path = sub_db_path + '/' + folder_name
            if not os.path.isdir(folder_path):
                continue
            # Get instrus and prs from a folder name
            pr0, instru0, T0, pr1, instru1, T1 = build_data_aux.get_instru_and_pr_from_fodler_path(folder_path, quantization)
            T += min(T0, T1)
            # Modify the mapping from instrument to indices in pianorolls and pitch bounds
            instrument_mapping = build_data_aux.instru_pitch_range(instrumentation=instru0,
                                                                   pr=pr0,
                                                                   instrument_mapping=instrument_mapping,
                                                                   instrument_list_from_dico=instrument_list_from_dico,
                                                                   )
            # Remark : instrument_mapping would be modified if it is only passed to the function,
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

    import pdb; pdb.set_trace()
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
    pr_orchestra = np.zeros((T, counter), dtype=np.int16)
    pr_piano = np.zeros((T, instrument_mapping['piano']['index_max']), dtype=np.int16)

    # Write the prs in the matrix
    time = 0
    change_track = []
    for sub_db in subfolder_names:
        sub_db_path = path_db + '/' + sub_db
        if not os.path.isdir(sub_db_path):
            continue
        for folder_name in os.listdir(sub_db_path):
            folder_path = sub_db_path + '/' + folder_name
            if not os.path.isdir(folder_path):
                continue
            # It's a new track ! Let's write its time index in change_track !!
            change_track.append(time)
            # Get instrus and prs from a folder name name
            pr0, instru0, T0, pr1, instru1, T1 = build_data_aux.get_instru_and_pr_from_fodler_path(folder_path, quantization)
            duration = min(T0, T1)
            # !! Time warping !! (tonnerre et tout)     :)
            # In fact just do a linear warping          :(
            if T0 > T1:
                pr0 = build_data_aux.warp_pr(pr0, T_source=T0, T_target=T1)
            elif T0 < T1:
                pr1 = build_data_aux.warp_pr(pr1, T_source=T1, T_target=T0)
            # Do nothing if T0 = T1

            # Find which pr is orchestra, which one is piano
            if len(set(instru0.keys())) > len(set(instru1.keys())):
                # Add the small pr to the general structure
                # pr0 is orchestra
                pr_orchestra = build_data_aux.cast_small_pr_into_big_pr(pr0, {}, time, duration, instrument_mapping, pr_orchestra)
                pr_piano = build_data_aux.cast_small_pr_into_big_pr(pr1, instru1, time, duration, instrument_mapping, pr_piano)
            elif len(set(instru0.keys())) < len(set(instru1.keys())):
                # pr1 is orchestra
                pr_piano = build_data_aux.cast_small_pr_into_big_pr(pr0, instru0, time, duration, instrument_mapping, pr_piano)
                pr_orchestra = build_data_aux.cast_small_pr_into_big_pr(pr1, {}, time, duration, instrument_mapping, pr_orchestra)
            else:
                raise Exception('The two midi files have the same number of instruments')
            time += duration
            del pr0, pr1, instru0, instru1

    # Save pr_orchestra, pr_piano, instrument_mapping
    data = {}
    data['quantization'] = quantization
    data['instru_mapping'] = instrument_mapping
    data['pr_orchestra'] = pr_orchestra
    data['pr_piano'] = pr_piano
    data['change_track'] = change_track
    pickle.dump(data, open(save_path, 'wb'))

if __name__ == '__main__':
    folder_path = '/Users/leo/Recherche/GitHub_Aciditeam/database/Orchestration/Orchestration_checked'
    subfolder_names = [
        'bouliane',
        'hand_picked_Spotify',
        'liszt_classical_archives'
    ]
    # subfolder_names = ['test']
    build_data(folder_path, subfolder_names, quantization=12, granularity='frame_level')
