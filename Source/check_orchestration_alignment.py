#!/usr/bin/env python
# -*- coding: utf8 -*-


import build_data_aux
import os
import numpy as np
from subprocess import call

from acidano.data_processing.utils.pianoroll_processing import sum_along_instru_dim, pitch_class
from acidano.data_processing.utils.time_warping import linear_warp_pr, dtw_pr, needleman_chord_wrapper, needleman_event_chord_wrapper
from acidano.data_processing.utils.event_level import get_event_ind
from acidano.data_processing.utils.pianoroll_processing import get_pianoroll_time
from acidano.visualization.numpy_array.dumped_numpy_to_csv import dump_to_csv
from acidano.data_processing.midi.write_midi import write_midi
# from acidano.data_processing.midi.read_midi import Read_midi


def check_orchestration_alignment(path_db, subfolder_names):
    for sub_db in subfolder_names:
        print '#' * 30
        print sub_db
        sub_db_path = path_db + '/' + sub_db
        if not os.path.isdir(sub_db_path):
            continue
        for folder_name in os.listdir(sub_db_path):

            ############################################################
            ############################################################
            ############################################################
            # folder_name = '1'
            ############################################################
            ############################################################
            ############################################################

            print '#' * 20
            print '#' + folder_name + '\n'
            folder_path = sub_db_path + '/' + folder_name
            if not os.path.isdir(folder_path):
                continue

            # Get instrus and prs from a folder name name
            pr0, instru0, T0, pr1, instru1, T1 = build_data_aux.get_instru_and_pr_from_folder_path(folder_path, quantization=12, clip=True)

            # !! Time warping !! (tonnerre et tout)     :)
            # In fact just do a linear warping          :(
            # Sum all instrument
            if T0 < T1:
                # pr0 = linear_warp_pr(pr0, T_target=T1)
                # pr0, pr1 = dtw_pr(pr0, pr1)
                # ppp = pitch_class(pr0)
                # trace0, trace1 = needleman_chord_wrapper(sum_along_instru_dim(pr0), sum_along_instru_dim(pr1))
                pr0, pr1 = needleman_event_chord_wrapper(pr0, pr1)
            elif T0 > T1:
                # pr1 = linear_warp_pr(pr1, T_target=T0)
                # ppp = pitch_class(pr0)
                # pr1, pr0 = dtw_pr(pr1, pr0)
                # pr0, pr1 = needleman_chord_wrapper(pr0, pr1)
                pr0, pr1 = needleman_event_chord_wrapper(pr0, pr1)
            # Do nothing if T0 = T1

            write_midi(pr=pr0, quantization=12, write_path='DEBUG/pr0.mid', tempo=80)
            write_midi(pr=pr1, quantization=12, write_path='DEBUG/pr1.mid', tempo=80)

            # Sum all instrument
            AAA = sum_along_instru_dim(pr0)
            BBB = sum_along_instru_dim(pr1)
            CCC = np.concatenate((AAA, BBB), axis=1)
            DDD = np.maximum(AAA, BBB)

            write_midi(pr={'piano': DDD}, quantization=12, write_path='DEBUG/FUSION.mid', tempo=80)

            np.savetxt('DEBUG/temp.csv', CCC, delimiter=',')
            dump_to_csv('DEBUG/temp.csv', 'DEBUG/temp.csv')

            call(["open", "DEBUG/numpy_vis.html"])

            # np.savetxt('DEBUG/temp.csv', CCC[-500:-1], delimiter=',')
            # dump_to_csv('DEBUG/temp.csv', 'DEBUG/temp.csv')
            #
            # call(["open", "DEBUG/numpy_vis.html"])

            import pdb; pdb.set_trace()


if __name__ == '__main__':
    folder_path = '/Users/leo/Recherche/GitHub_Aciditeam/database/Orchestration/Orchestration_checked'
    subfolder_names = [
        'liszt_classical_archives',
        'bouliane',
        'hand_picked_Spotify',
    ]
    check_orchestration_alignment(folder_path, subfolder_names)
