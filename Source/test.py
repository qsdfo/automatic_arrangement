#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import build_data_aux
from acidano.visualization.numpy_array.dumped_numpy_to_csv import dump_to_csv
from acidano.data_processing.utils.event_level import get_event_ind
from acidano.data_processing.utils.pianoroll_processing import sum_along_instru_dim


if __name__ == '__main__':

    folder_path = "/Users/leo/Recherche/GitHub_Aciditeam/database/Orchestration/Orchestration_checked/liszt_classical_archives/0"
    quantization = 12
    pr0, instru0, T0, pr1, instru1, T1 = build_data_aux.get_instru_and_pr_from_folder_path(folder_path, quantization=quantization, clip=True)

    pr0_A = sum_along_instru_dim(pr0)
    pr1_A = sum_along_instru_dim(pr1)

    el_0 = np.concatenate((np.zeros(1, dtype=np.int16), get_event_ind(pr0_A)))
    el_1 = np.concatenate((np.zeros(1, dtype=np.int16), get_event_ind(pr1_A)))

    import pdb; pdb.set_trace()

    print(el_0[:20])
    print(el_1[:20])

    np.savetxt('DEBUG/0.csv', pr0_A[:500, :], delimiter=',')
    dump_to_csv('DEBUG/0.csv', 'DEBUG/0.csv')

    np.savetxt('DEBUG/1.csv', pr1_A[:500, :], delimiter=',')
    dump_to_csv('DEBUG/1.csv', 'DEBUG/1.csv')

    pr0_red = pr0_A[el_0,:]
    pr1_red = pr1_A[el_1,:]

    np.savetxt('DEBUG/R0.csv', pr0_red[:200, :], delimiter=',')
    dump_to_csv('DEBUG/R0.csv', 'DEBUG/R0.csv')

    np.savetxt('DEBUG/R1.csv', pr1_red[:200, :], delimiter=',')
    dump_to_csv('DEBUG/R1.csv', 'DEBUG/R1.csv')
