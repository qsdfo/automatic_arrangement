#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import re
import os

def load_matrices(chunk_path_list, parameters):
    """Input : 
        - chunk_path_list : list of splitted matrices to be concatenated

    This function build the matrix corresponing to a coherent ensemble of files, for example only train files
    """
    tt=0
    T_max = len(chunk_path_list)*parameters["chunk_size"] 
    orch_transformed = np.zeros((T_max, parameters["N_orchestra"]), dtype=np.float16)
    piano_transformed = np.zeros((T_max, parameters["N_piano"]), dtype=np.float16)
    piano_embedded = np.zeros((T_max, parameters["N_piano_embedded"]), dtype=np.float16)
    
    for block_folder in chunk_path_list:
        pr_piano_transformed, pr_piano_embedded, pr_orch_transformed = load_matrix_NO_PROCESSING(block_folder, parameters['duration_piano'])
        length = pr_piano_transformed.shape[0]
        piano_transformed[tt:tt+length]=pr_piano_transformed
        piano_embedded[tt:tt+length]=pr_piano_embedded
        orch_transformed[tt:tt+length]=pr_orch_transformed
        tt += length

    return piano_transformed, piano_embedded, orch_transformed

def load_matrix_NO_PROCESSING(block_folder, duration_piano_bool):
    piano_file = os.path.join(block_folder, 'pr_piano_transformed.npy')
    orch_file = re.sub('piano', 'orch', piano_file)
    piano_embedded_file = re.sub('piano_transformed', 'piano_embedded', piano_file)
    pr_piano_transformed = np.load(piano_file)
    pr_piano_embedded = np.load(piano_embedded_file)
    pr_orch_transformed = np.load(orch_file)

    # if parameters['mask_orch']:
    #     mask_orch_file = re.sub('piano', 'mask_orch', piano_file)
    #     mask_orch = np.load(mask_orch_file)
    # else:
    #     mask_orch = None

    return pr_piano_transformed, pr_piano_embedded, pr_orch_transformed