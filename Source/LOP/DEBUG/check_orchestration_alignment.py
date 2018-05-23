#!/usr/bin/env python
# -*- coding: utf8 -*-


import build_data_aux
import os
import numpy as np

from acidano.data_processing.utils.pianoroll_processing import sum_along_instru_dim
from acidano.visualization.numpy_array.visualize_numpy import visualize_mat
from acidano.data_processing.midi.write_midi import write_midi

# from acidano.data_processing.utils.time_warping import needleman_chord_wrapper, warp_dictionnary_trace, remove_zero_in_trace, warp_pr_aux
# from acidano.data_processing.utils.event_level import get_event_ind_dict
# from acidano.data_processing.midi.read_midi import Read_midi
# import acidano.data_processing.utils.unit_type as Unit_type


def check_zero_orchestra():
    context_size = 100
    data_folder = '../Data'
    set_identifier = 'train'
    # Load data
    piano = np.load(data_folder + '/piano_' + set_identifier + '.csv')
    orch = np.load(data_folder + '/orchestra_' + set_identifier + '.csv')

    # Detect problems
    flat_orch = orch.sum(axis=1)
    flat_piano = piano.sum(axis=1)
    ind_problems = np.where((flat_orch == 0) ^ (flat_piano == 0))[0]

    for t in ind_problems:
        min_context = max(0, t-context_size)
        max_context = min(piano.shape[0], t+context_size)
        zeros_mat = np.zeros((max_context-min_context, 30))
        zeros_mat[:, 15] = 1
        context_piano = piano[min_context:max_context]
        context_orch = orch[min_context:max_context]
        plot_mat = np.concatenate((context_piano, zeros_mat, context_orch), axis=1)
        # Plot around the problem
        visualize_mat(plot_mat, 'DEBUG/plot_problems', str(t))
    return 0


def check_orchestration_alignment(path_db, subfolder_names, temporal_granularity, quantization, unit_type, gapopen, gapextend):

    output_dir = 'DEBUG/' +\
                 'Grid_search_database_alignment/' + str(quantization) +\
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
            print '# ' + sub_db + ' : ' + folder_name
            folder_path = sub_db_path + '/' + folder_name
            if not os.path.isdir(folder_path):
                continue

            # Skip already computed folders
            save_folder_name = output_dir +\
                '/' + sub_db + '_' + folder_name
            if os.path.isdir(save_folder_name):
                continue

            pr_piano_no_map, _, _, _, _, pr_orchestra_no_map, _, _, instru_orch, _, duration =\
                build_data_aux.process_folder(folder_path, quantization, binary_piano, binary_orch, temporal_granularity, gapopen, gapextend)

            # Apply the mapping
            pr_piano = {}
            pr_orchestra = {}
            for k, v in pr_piano_no_map.iteritems():
                if 'Piano' in pr_piano:
                    pr_piano['Piano'] = np.maximum(pr_piano['Piano'], v)
                else:
                    pr_piano['Piano'] = v

            for k, v in pr_orchestra_no_map.iteritems():
                # unmix instrus
                new_k = instru_orch[k.rstrip('\x00')]
                instru_names = build_data_aux.unmixed_instru(new_k)
                for instru_name in instru_names:
                    if instru_name in pr_orchestra:
                        pr_orchestra[instru_name] = np.maximum(pr_orchestra[instru_name], v)
                    else:
                        pr_orchestra[instru_name] = v

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

            if not os.path.isdir(save_folder_name):
                os.makedirs(save_folder_name)

            visualize_mat(CCC_aligned, save_folder_name, 'aligned')
            # write_midi(pr={'piano1': sum_along_instru_dim(pr0)}, quantization=quantization, write_path=save_folder_name + '/0.mid', tempo=80)
            # write_midi(pr={'piano1': sum_along_instru_dim(pr1)}, quantization=quantization, write_path=save_folder_name + '/1.mid', tempo=80)
            write_midi(pr=pr_piano, quantization=quantization_write, write_path=save_folder_name + '/0.mid', tempo=80)
            write_midi(pr=pr_orchestra, quantization=quantization_write, write_path=save_folder_name + '/1.mid', tempo=80)
            write_midi(pr={'Piano': piano_aligned, 'Violin': orchestra_aligned}, quantization=quantization_write, write_path=save_folder_name + '/both_aligned.mid', tempo=80)
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
    folder_path = '/Users/leo/Recherche/GitHub_Aciditeam/database/Orchestration/LOP_database_29_05_17'
    subfolder_names = [
        'bouliane',
        'imslp',
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
