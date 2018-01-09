#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import re
import cPickle as pkl
from LOP.Utils.process_data import process_data_piano, process_data_orch

def load_matrices(piano_file, parameters):
    orch_file = re.sub('piano', 'orchestra', piano_file)
    piano = np.load(piano_file)
    orch = np.load(orch_file)
    if parameters['duration_piano']:
        duration_piano_file = re.sub('piano', 'duration_piano', piano_file)
        duration_piano = np.load(duration_piano_file)
    else:
        duration_piano = None
        
    if parameters['mask_orch']:
        mask_orch_file = re.sub('piano', 'mask_orch', piano_file)
        mask_orch = np.load(mask_orch_file)
    else:
        mask_orch = None

    piano = process_data_piano(piano, duration_piano, parameters)
    orch = process_data_orch(orch, parameters)

    # ####################################################
    # ####################################################
    # # TEMP : plot random parts of the data to check alignment
    # from LOP_database.visualization.numpy_array.visualize_numpy import visualize_mat
    # T = len(piano)
    # for t in np.arange(100, T, 1000):
    #   AAA = np.concatenate((piano[t-20:t]*2, orch[t-20:t]), axis=1)
    #   visualize_mat(AAA, "debug", str(t))
    # ####################################################
    # ####################################################
    tracks_start_end_file = re.sub(r'piano(.*)\.npy$', r'tracks_start_end\1.pkl', piano_file)
    tracks_start_end = pkl.load(open(tracks_start_end_file))
    return piano, orch, duration_piano, mask_orch, tracks_start_end