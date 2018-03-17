#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import re
import os
import cPickle as pkl
from LOP.Utils.process_data import process_data_piano, process_data_orch
import time

# def load_matrices(piano_file, parameters):
#     orch_file = re.sub('piano', 'orchestra', piano_file)
#     piano = np.load(piano_file)
#     orch = np.load(orch_file)
#     if parameters['duration_piano']:
#         duration_piano_file = re.sub('piano', 'duration_piano', piano_file)
#         duration_piano = np.load(duration_piano_file)
#     else:
#         duration_piano = None
        
#     if parameters['mask_orch']:
#         mask_orch_file = re.sub('piano', 'mask_orch', piano_file)
#         mask_orch = np.load(mask_orch_file)
#     else:
#         mask_orch = None

#     piano = process_data_piano(piano, duration_piano, parameters)
#     orch = process_data_orch(orch, parameters)

#     # ####################################################
#     # ####################################################
#     # # TEMP : plot random parts of the data to check alignment
#     # from LOP_database.visualization.numpy_array.visualize_numpy import visualize_mat
#     # T = len(piano)
#     # for t in np.arange(100, T, 1000):
#     #   AAA = np.concatenate((piano[t-20:t]*2, orch[t-20:t]), axis=1)
#     #   visualize_mat(AAA, "debug", str(t))
#     # ####################################################
#     # ####################################################

#     tracks_start_end_file = re.sub(r'piano(.*)\.npy$', r'tracks_start_end\1.pkl', piano_file)
#     tracks_start_end = pkl.load(open(tracks_start_end_file))
    
#     return piano, orch, duration_piano, mask_orch, tracks_start_end

def load_matrices(chunk_path_list, parameters):
    """Input : 
        - chunk_path_list : list of splitted matrices to be concatenated

    This function build the matrix corresponing to a coherent ensemble of files, for example only train files
    """
    time=0
    T_max = len(chunk_path_list)*parameters["chunk_size"] 
    orch = np.zeros((T_max, parameters["N_orchestra"]), dtype=np.float16)
    piano = np.zeros((T_max, parameters["N_piano"]), dtype=np.float16)
    if parameters['duration_piano']:
        duration_piano = np.zeros((T_max, 1))
    else:
        duration_piano = None

    for block_folder in chunk_path_list:
        pr_piano, pr_orch, this_duration = load_matrix_NO_PROCESSING(block_folder, parameters['duration_piano'])
        length = pr_piano.shape[0]
        piano[time:time+length]=pr_piano
        orch[time:time+length]=pr_orch
        if parameters['duration_piano']:
            duration_piano[time:time+length] = this_duration
        time += length

    piano = process_data_piano(piano, duration_piano, parameters)
    orch = process_data_orch(orch, parameters)

    ##################################################
    ##################################################
    ##################################################
    # Data augmentation ici ?
    duration_piano = None
    mask_orch = None 
    ##################################################
    ##################################################
    ##################################################

    return piano, orch, duration_piano, mask_orch

def load_matrix_NO_PROCESSING(block_folder, duration_piano_bool):
    piano_file = os.path.join(block_folder, 'pr_piano.npy')
    orch_file = re.sub('piano', 'orch', piano_file)
    pr_piano = np.load(piano_file)
    pr_orch = np.load(orch_file)
    
    if duration_piano_bool:
        duration_piano_file = re.sub('piano', 'duration_piano', piano_file)
        this_duration = np.load(duration_piano_file)
    else:
        this_duration = None

    # if parameters['mask_orch']:
    #     mask_orch_file = re.sub('piano', 'mask_orch', piano_file)
    #     mask_orch = np.load(mask_orch_file)
    # else:
    #     mask_orch = None

    return pr_piano, pr_orch, this_duration