#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import cPickle
import theano
import math

from acidano.data_processing.minibatch_builder import k_fold_cross_validation, tvt_minibatch, tvt_minibatch_seq
from acidano.data_processing.event_level import get_event_ind


def get_data(data_path, temporal_granularity, temporal_order, sequential_learning=False, shared_bool=True, bin_unit_bool=True):
    """
    Load data from pickle (.p) file into a matrix. Return valid indexes for sequential learning

    data_path:      relative path to pickelized data
    temporal_order: temporal order of the model for which data are loaded

    data :          shared variable containing the pianoroll representation of all the file concatenated in
                    a single huge pr
    valide_indexes: list of valid training point given the temporal order

    """

    data = cPickle.load(open(data_path, "rb"))

    # Unpack matrices
    pr_orchestra = data['pr_orchestra']
    pr_piano = data['pr_piano']
    new_track_ind = data['change_track']

    # Build valid indices for building minibatches
    pad_indices = []
    T = pr_orchestra.shape[0]
    l1 = new_track_ind[:]  # [:] = copy by value
    l2 = new_track_ind[:]
    l2.append(T)
    l2.pop(0)
    # Two cases :
    if sequential_learning:
        # RNN-like models: chunk of size temporal_order, overlap 50%
        for s, e in zip(l1, l2):
            step = math.floor(temporal_order / 2)
            for i in range(s + temporal_order, e, step):
                pad_indices.append(i)
    else:
        # others
        for s, e in zip(l1, l2):
            pad_indices.extend(range(s + temporal_order, e))

    # Temporal granularity
    if temporal_granularity == 'frame_level':
        valid_indices = pad_indices
    elif temporal_granularity == 'event_level':
        new_event_ind = get_event_ind(pr_orchestra)
        # Valid indices are the intersection of new event and valid
        # Note that this operation shuffle the indices
        valid_indices = set(new_event_ind).intersection(pad_indices)
    elif temporal_granularity == 'full_event_level':
        new_event_ind = get_event_ind(pr_orchestra)
        # Reduce pr
        pr_orchestra = pr_orchestra[new_event_ind, :]
        pr_piano = pr_piano[new_event_ind, :]
        # Compute valid indices
        l3 = new_track_ind[:]
        l3.append(T)
        valid_indices = full_event_level_ind(l3, new_event_ind, temporal_order)

    # Cast valid_index in a numpy array
    valid_indices = np.array(valid_indices)

    # Binary representation
    if bin_unit_bool:
        pr_orchestra = (pr_orchestra > 0).astype(int)
        pr_piano = (pr_piano > 0).astype(int)

    if shared_bool:
        # Instanciate shared variables
        orch_shared = theano.shared(np.asarray(pr_orchestra, dtype=theano.config.floatX))
        piano_shared = theano.shared(np.asarray(pr_piano, dtype=theano.config.floatX))
    else:
        orch_shared = pr_orchestra
        piano_shared = pr_piano

    return orch_shared, piano_shared, valid_indices


def full_event_level_ind(new_track_ind, new_event_ind, temporal_order):
    m = 0
    n = 0
    counter = 0
    out_list = []
    for elem in new_event_ind:
        if new_track_ind[n] <= elem:
            n += 1
            counter = 1
        else:
            if counter >= temporal_order:
                out_list.append(m)
            counter += 1
        m += 1
    return out_list


def load_data_k_fold(data_path, temporal_granularity, temporal_order, shared_bool, bin_unit_bool, minibatch_size, split=(0.7, 0.1, 0.2)):
    orch, piano, valid_index = get_data(data_path, temporal_granularity, temporal_order, False, shared_bool, bin_unit_bool)
    train_index, validate_index, test_index = k_fold_cross_validation(valid_index, minibatch_size, split)
    # train_index, validate_index, test_index = tvt_minibatch(log_file_path, valid_index, minibatch_size, shuffle, split)
    return orch, piano, train_index, validate_index, test_index


def load_data_tvt(data_path, temporal_granularity, temporal_order, shared_bool, bin_unit_bool, minibatch_size, split=(0.7, 0.1, 0.2)):
    orch, piano, valid_index = get_data(data_path, temporal_granularity, temporal_order, False, shared_bool, bin_unit_bool)
    # train_index, validate_index, test_index = k_fold_cross_validation(log_file_path, valid_index, minibatch_size, split)
    train_index, validate_index, test_index = tvt_minibatch(valid_index, minibatch_size, 'block', split)
    return orch, piano, train_index, validate_index, test_index


def load_data_seq_tvt(data_path, temporal_granularity, temporal_order, shared_bool, bin_unit_bool, split=(0.7, 0.1, 0.2)):
    # In this case minibatch_size is equal to temporal_granularity
    orch, piano, valid_index = get_data(data_path, temporal_granularity, temporal_order, True, shared_bool, bin_unit_bool)
    train_index, validate_index, test_index = tvt_minibatch_seq(valid_index, temporal_order, True, split)
    return orch, piano, train_index, validate_index, test_index


if __name__ == '__main__':
    orch, orch_mapping, piano, piano_mapping, \
        train_batch_ind, validate_batch_ind, test_batch_ind =\
        load_data_tvt(data_path='../Data/data.p',
                      temporal_granularity='full_event_level',
                      temporal_order=8,
                      shared_bool=True,
                      bin_unit_bool=True,
                      minibatch_size=100)
