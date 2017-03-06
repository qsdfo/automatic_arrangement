#!/usr/bin/env python
# -*- coding: utf8 -*-


import build_data_aux
import os
import numpy as np

import itertools
import build_data_aux

from acidano.data_processing.utils.pianoroll_processing import sum_along_instru_dim
from acidano.data_processing.utils.time_warping import needleman_chord_wrapper, warp_dictionnary_trace, remove_zero_in_trace, warp_pr_aux
from acidano.visualization.numpy_array.visualize_numpy import visualize_mat
from acidano.data_processing.midi.write_midi import write_midi
from acidano.data_processing.utils.event_level import get_event_ind_dict
# from acidano.data_processing.midi.read_midi import Read_midi

import acidano.data_processing.utils.unit_type as Unit_type


def check_orchestration_alignment(path_db, subfolder_names, temporal_granularity, quantization, unit_type, gapopen, gapextend):

    output_dir = 'Grid_search_database_alignment/' + str(quantization) +\
                 '_' + temporal_granularity +\
                 '_' + unit_type +\
                 '_' + str(gapopen) +\
                 '_' + str(gapextend)

    if temporal_granularity == "event_level":
        quantization_write = 1
    else:
        quantization_write = quantization

    counter = 0
    sum_score = 0
    nbFrame = 0
    nbId = 0
    nbDiffs = 0

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

            pr_piano, instru_piano, name_piano, pr_orchestra, instru_orchestra, name_orchestra, duration =\
                build_data_aux.process_folder(folder_path, quantization, unit_type, temporal_granularity, None, gapopen, gapextend)

            if duration is None:
                continue

            # Sum all instrument
            piano_aligned = sum_along_instru_dim(pr_piano)
            orchestra_aligned = sum_along_instru_dim(pr_orchestra)
            OOO_aligned = np.zeros((duration, 30), dtype=np.int16)
            CCC_aligned = np.concatenate((piano_aligned, OOO_aligned, orchestra_aligned), axis=1)

            # Update statistics
            # nbFrame += duration
            # sum_score += this_sum_score
            # nbId += this_nbId
            # nbDiffs += this_nbDiffs

            # counter = counter + 1

            # Save every 100 example
            # if counter % 10 == 0:
            #     import pdb; pdb.set_trace()

            save_folder_name = output_dir +\
                '/' + sub_db + '_' + folder_name

            if not os.path.isdir(save_folder_name):
                os.makedirs(save_folder_name)

            visualize_mat(CCC_aligned, save_folder_name, 'aligned')
            # write_midi(pr={'piano1': sum_along_instru_dim(pr0)}, quantization=quantization, write_path=save_folder_name + '/0.mid', tempo=80)
            # write_midi(pr={'piano1': sum_along_instru_dim(pr1)}, quantization=quantization, write_path=save_folder_name + '/1.mid', tempo=80)
            write_midi(pr=pr_piano, quantization=quantization_write, write_path=save_folder_name + '/0.mid', tempo=80)
            write_midi(pr=pr_orchestra, quantization=quantization_write, write_path=save_folder_name + '/1.mid', tempo=80)
            write_midi(pr={'piano': piano_aligned, 'violin': orchestra_aligned}, quantization=quantization_write, write_path=save_folder_name + '/both_aligned.mid', tempo=80)
            # write_midi(pr={'piano1': AAA_aligned, 'piano2': BBB_aligned}, quantization=quantization_write, write_path=save_folder_name + '/both__aligned.mid', tempo=80)

    # # Write statistics
    # mean_score = float(sum_score) / max(1,nbFrame)
    # nbId_norm = nbId / quantization
    # nbDiffs_norm = nbDiffs / quantizationfolder_path
    #             "quantization = %d\n" % quantization +
    #             "Gapopen = %d\n" % gapopen +
    #             "Gapextend = %d\n" % gapextend +
    #             "Number frame = %d\n" % nbFrame +
    #             "\n\n\n" +
    #             "Sum score = %d\n" % sum_score+
    #             "Mean score = %f\n" % mean_score+
    #             "Number id = %d\n" % nbId +
    #             "Number id / quantization = %d\n" % nbId_norm+
    #             "Number diffs = %d\n" % nbDiffs+
    #             "Number diffs / quantization = %d\n" % nbDiffs_norm)


if __name__ == '__main__':
    folder_path = '../../database/Orchestration/Orchestration_checked'
    subfolder_names = [
        'bouliane',
        'hand_picked_Spotify',
        'liszt_classical_archives',
    ]

    # grid_search = {}
    # grid_search['gapopen'] = [3]
    # # grid_search['quantization'] = [4, 8, 12]
    # grid_search['quantization'] = [100]
    #
    # # Build all possible values
    # for gapopen, quantization in list(itertools.product(
    #     grid_search['gapopen'],
    #     grid_search['quantization'])
    # ):
    #     for gapextend in range(gapopen):
    #         check_orchestration_alignment(folder_path, subfolder_names, 'event_level', quantization, 'binary', gapopen, gapextend)

    check_orchestration_alignment(folder_path, subfolder_names, 'event_level', 100, 'binary', 3, 1)
