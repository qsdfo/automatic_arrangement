#!/usr/bin/env python
# -*- coding: utf8 -*-


import build_data_aux
import os
import numpy as np

import itertools

from acidano.data_processing.utils.pianoroll_processing import sum_along_instru_dim
from acidano.data_processing.utils.time_warping import needleman_chord_wrapper, warp_dictionnary_trace, remove_zero_in_trace
from acidano.visualization.numpy_array.visualize_numpy import visualize_mat
from acidano.data_processing.midi.write_midi import write_midi
# from acidano.data_processing.midi.read_midi import Read_midi


def check_orchestration_alignment(path_db, subfolder_names, quantization, gapopen, gapextend):

    output_dir = 'DEBUG/' + str(quantization) +\
                 '_' + str(gapopen) +\
                 '_' + str(gapextend)

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

            # Get instrus and prs from a folder name name
            pr0, instru0, T0, path_0, pr1, instru1, T1, path_1 = build_data_aux.get_instru_and_pr_from_folder_path(folder_path, quantization=quantization, clip=True)

            # Get trace from needleman_wunsch algorithm
            # Traces are binary lists, 0 meaning a gap is inserted
            trace_0, trace_1, this_sum_score, this_nbId, this_nbDiffs = needleman_chord_wrapper(sum_along_instru_dim(pr0), sum_along_instru_dim(pr1), gapopen, gapextend)

            # Warp dictionnaries according to the traces
            assert(len(trace_0) == len(trace_1)), "size mismatch"
            pr0_warp = warp_dictionnary_trace(pr0, trace_0)
            pr1_warp = warp_dictionnary_trace(pr1, trace_1)

            # In fact we just discard 0 in traces for both pr
            trace_prod = [e1 * e2 for (e1,e2) in zip(trace_0, trace_1)]
            if sum(trace_prod) == 0:
                # It's definitely not a match...
                # Check for the files : are they really an piano score and its orchestration ??
                with(open('log.txt', 'a')) as f:
                    f.write(folder_path + '\n')
                continue
            pr0_aligned = remove_zero_in_trace(pr0_warp, trace_prod)
            pr1_aligned = remove_zero_in_trace(pr1_warp, trace_prod)

            # Sum all instrument
            AAA_warp = sum_along_instru_dim(pr0_warp)
            BBB_warp = sum_along_instru_dim(pr1_warp)
            OOO_warp = np.zeros((BBB_warp.shape[0], 30), dtype=np.int16)
            CCC_warp = np.concatenate((AAA_warp, OOO_warp, BBB_warp), axis=1)
            AAA_aligned = sum_along_instru_dim(pr0_aligned)
            BBB_aligned = sum_along_instru_dim(pr1_aligned)
            OOO_aligned = np.zeros((BBB_aligned.shape[0], 30), dtype=np.int16)
            CCC_aligned = np.concatenate((AAA_aligned, OOO_aligned, BBB_aligned), axis=1)

            # Update statistics
            nbFrame += len(trace_0)
            sum_score += this_sum_score
            nbId += this_nbId
            nbDiffs += this_nbDiffs

            counter = counter + 1

            # Save every 100 example
            # if counter % 10 == 0:
            #     import pdb; pdb.set_trace()

            save_folder_name = output_dir +\
                '/' + sub_db + '_' + folder_name

            if not os.path.exists(save_folder_name):
                os.makedirs(save_folder_name)

            visualize_mat(CCC_warp, save_folder_name, 'warp')
            visualize_mat(CCC_aligned, save_folder_name, 'aligned')
            # write_midi(pr={'piano1': sum_along_instru_dim(pr0)}, quantization=quantization, write_path=save_folder_name + '/0.mid', tempo=80)
            # write_midi(pr={'piano1': sum_along_instru_dim(pr1)}, quantization=quantization, write_path=save_folder_name + '/1.mid', tempo=80)
            write_midi(pr=pr0, quantization=quantization, write_path=save_folder_name + '/0.mid', tempo=80)
            write_midi(pr=pr1, quantization=quantization, write_path=save_folder_name + '/1.mid', tempo=80)
            write_midi(pr={'piano1': AAA_warp, 'piano2': BBB_warp}, quantization=quantization, write_path=save_folder_name + '/both__warp.mid', tempo=80)
            write_midi(pr={'piano1': AAA_aligned, 'piano2': BBB_aligned}, quantization=quantization, write_path=save_folder_name + '/both__aligned.mid', tempo=80)

    # Write statistics
    mean_score = float(sum_score) / max(1,nbFrame)
    nbId_norm = nbId / quantization
    nbDiffs_norm = nbDiffs / quantization

    with open(output_dir + '/log.txt', 'wb') as f:
        f.write("##########################\n" +
                "quantization = %d\n" % quantization +
                "Gapopen = %d\n" % gapopen +
                "Gapextend = %d\n" % gapextend +
                "Number frame = %d\n" % nbFrame +
                "\n\n\n" +
                "Sum score = %d\n" % sum_score+
                "Mean score = %f\n" % mean_score+
                "Number id = %d\n" % nbId +
                "Number id / quantization = %d\n" % nbId_norm+
                "Number diffs = %d\n" % nbDiffs+
                "Number diffs / quantization = %d\n" % nbDiffs_norm)


if __name__ == '__main__':
    folder_path = '../../database/Orchestration/Orchestration_checked'
    subfolder_names = [
        'bouliane',
        'hand_picked_Spotify',
        'liszt_classical_archives',
    ]

    grid_search = {}
    # grid_search['gapopen'] = [1, 2, 3, 4, 5]
    # grid_search['quantization'] = [4, 8, 12]
    grid_search['gapopen'] = [1]
    grid_search['quantization'] = [4]

    # Build all possible values
    for gapopen, quantization in list(itertools.product(
        grid_search['gapopen'],
        grid_search['quantization'])
    ):
        for gapextend in range(gapopen):
            check_orchestration_alignment(folder_path, subfolder_names, quantization, gapopen, gapextend)
