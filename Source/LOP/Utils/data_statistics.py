#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Module for collecting statistics on the training data in order to perform pre-processing
Created on Mon Dec  4 16:30:20 2017

@author: leo
"""

import numpy as np
import re
from load_matrices import load_matrices

def get_activation_ratio(train_folds, orch_dim, parameters):
    num_activation = np.zeros((orch_dim))
    num_notes = 0
    # Compute statistique on each chunk
    for chunk in train_folds:
        _, orch, _, _ = load_matrices(chunk["chunks_folders"], parameters)
        num_activation += np.sum(orch>0, axis=0)
        num_notes += float(orch.shape[0])
    ratio_activation = num_activation / num_notes
    return ratio_activation

def compute_static_bias_initialization(ratio_activation, epsilon=1e-5):
    ratio_activation = np.maximum(ratio_activation, epsilon)
    # Inverse sigmoid !
    static_bias = np.log(ratio_activation/ (1-ratio_activation))
    return static_bias

def get_mask_inter_orch_NADE(train_folds, orch_dim, parameters):
    mask_inter_orch = np.zeros((orch_dim, orch_dim))
    num_co_occurences = np.zeros((orch_dim, orch_dim))
    num_time_frames = np.zeros((orch_dim))
    # Compute statistique on each chunk
    for chunk in train_folds:
        _, orch, _, _ = load_matrices(chunk["chunks_folders"], parameters)
        for target_note in range(orch_dim):
            # Make sure it's binary
            frames_on = orch[:, target_note]>0
            masked_out_orch = orch[frames_on]
            
            num_time_frames[target_note] += frames_on.sum()
            num_co_occurences[target_note, :] += masked_out_orch.sum(axis=0)
    
    for target_note in range(orch_dim):
        if num_time_frames[target_note]==0:
            mask_inter_orch[target_note, :] = np.ones((orch_dim))
        else:
            mask_inter_orch[target_note, :] = num_co_occurences[target_note, :] / num_time_frames[target_note]
            
    return mask_inter_orch

def get_mask_piano_orch_NADE(train_folds, piano_dim, orch_dim, parameters):
    mask_inter_orch = np.zeros((piano_dim, orch_dim))
    num_co_occurences = np.zeros((piano_dim, orch_dim))
    num_time_frames = np.zeros((orch_dim))
    # Compute statistique on each chunk
    for chunk in train_folds:
        piano, orch, _, _ = load_matrices(chunk["chunks_folders"], parameters)
        for target_note in range(piano_dim):
            # Make sure it's binary
            frames_on = piano[:, target_note]>0
            masked_out_orch = orch[frames_on]
            
            num_time_frames[target_note] += frames_on.sum()
            num_co_occurences[target_note, :] += masked_out_orch.sum(axis=0)
    
    for target_note in range(piano_dim):
        if num_time_frames[target_note]==0:
            mask_inter_orch[target_note, :] = np.ones((orch_dim))
        else:
            mask_inter_orch[target_note, :] = num_co_occurences[target_note, :] / num_time_frames[target_note]

    return mask_inter_orch


def get_mean_number_units_on(train_folds, parameters):
    num_notes_on = []
    # Compute statistique on each chunk
    for chunk in train_folds:
        _, orch, _, _ = load_matrices(chunk["chunks_folders"], parameters)
        this_num_notes_on = np.sum(orch>0, axis=1)
        num_notes_on.extend(this_num_notes_on)
    mean_number_on = sum(num_notes_on) / float(len(num_notes_on))
    return mean_number_on