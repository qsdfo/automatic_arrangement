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
    for piano_path, train_batches in train_folds.iteritems():
        orch_train = extract_training_points(piano_path, train_batches, parameters)
        num_activation += np.sum(orch_train>0, axis=0)
        num_notes += float(orch_train.shape[0])
    ratio_activation = num_activation / num_notes
    return ratio_activation

def compute_static_bias_initialization(ratio_activation, epsilon=1e-5):
    ratio_activation = np.maximum(ratio_activation, epsilon)
    # Inverse sigmoid !
    static_bias = np.log(ratio_activation/ (1-ratio_activation))
    return static_bias

def get_mean_number_units_on(train_folds, parameters):
    num_notes_on = []
    for piano_path, train_batches in train_folds.iteritems():
        orch = extract_training_points(piano_path, train_batches, parameters)
        this_num_notes_on = np.sum(orch>0, axis=1)
        num_notes_on.extend(this_num_notes_on)
    mean_number_on = sum(num_notes_on) / float(len(num_notes_on))
    return mean_number_on

def extract_training_points(piano_path, batches, parameters):
    # Extract only training points
    piano, orch, _, _, _ = load_matrices(piano_path, parameters)
    flat_indices = [ind for batch in batches for ind in batch]
    return orch[flat_indices]