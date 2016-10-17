#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import theano

import logging
import random
import cPickle as pickle


def load_data(temporal_order=20, batch_size=100, generation_length=100,
              binary_unit=True, skip_sample=1,logger_load=None):
    # If no logger, create one
    if logger_load is None:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename='load.log',
                            filemode='w')
        logger_load=logging.getLogger('load')

    piano_train = np.load('../Data/piano_train.csv')
    orchestra_train = np.load('../Data/orchestra_train.csv')
    piano_valid = np.load('../Data/piano_valid.csv')
    orchestra_valid = np.load('../Data/orchestra_valid.csv')
    piano_test = np.load('../Data/piano_test.csv')
    orchestra_test = np.load('../Data/orchestra_test.csv')

    # Binary unit ?
    if binary_unit:
        piano_train[np.nonzero(piano_train)] = 1
        orchestra_train[np.nonzero(orchestra_train)] = 1
        piano_valid[np.nonzero(piano_valid)] = 1
        orchestra_valid[np.nonzero(orchestra_valid)] = 1
        piano_test[np.nonzero(piano_test)] = 1
        orchestra_test[np.nonzero(orchestra_test)] = 1

    # Shared variables : push data on GPU, memory problem for this dataset ??
    # First check type
    type_data = (piano_train.dtype != theano.config.floatX)
    if type_data:
        logger_load.warning('Incorrect data type for pianorolls')
        logger_load.warning(str(piano_train.dtype))
    # borrow=True avoid copying the entire matrix
    piano_train_shared = theano.shared(piano_train, name='piano_train', borrow=True)
    orchestra_train_shared = theano.shared(orchestra_train, name='orchestra_train', borrow=True)
    piano_valid_shared = theano.shared(piano_valid, name='piano_valid', borrow=True)
    orchestra_valid_shared = theano.shared(orchestra_valid, name='orchestra_valid', borrow=True)
    piano_test_shared = theano.shared(piano_test, name='piano_test', borrow=True)
    orchestra_test_shared = theano.shared(orchestra_test, name='orchestra_test', borrow=True)

    tracks_start_end_train = pickle.load(open('../Data/tracks_start_end_train.pkl', 'rb'))
    tracks_start_end_valid = pickle.load(open('../Data/tracks_start_end_valid.pkl', 'rb'))
    tracks_start_end_test = pickle.load(open('../Data/tracks_start_end_test.pkl', 'rb'))

    # Get valid indices given start_track and temporal_order
    def valid_indices(tracks_start_end, temporal_order):
        valid_ind = []
        for (start_track, end_track) in tracks_start_end.values():
            valid_ind.extend(range(start_track+temporal_order-1, end_track, skip_sample))
        return valid_ind

    def last_indices(tracks_start_end, temporal_order):
        valid_ind = []
        for (start_track, end_track) in tracks_start_end.values():
            # If the middle of the track is more than temporal_order,
            # Then store it as a generation index
            # if not, take the last index
            # If last index is still not enough, just skip the track
            half_duration = (end_track-start_track) / 2
            middle_track = start_track + half_duration
            if half_duration > temporal_order:
                valid_ind.append(middle_track)
            elif (end_track-start_track) > temporal_order:
                valid_ind.append(end_track-1)
        return valid_ind

    def build_batches(valid_ind):
        batches = []
        position = 0
        n_batch = int(len(valid_ind) // batch_size)

        # Shuffle indices
        random.shuffle(valid_ind)

        for i in range(n_batch):
            batches.append(valid_ind[position:position+batch_size])
            position += batch_size
        return batches

    train_index = valid_indices(tracks_start_end_train, temporal_order)
    train_batches = build_batches(train_index)

    valid_index = valid_indices(tracks_start_end_valid, temporal_order)
    valid_batches = build_batches(valid_index)

    test_index = valid_indices(tracks_start_end_test, temporal_order)
    test_batches = build_batches(test_index)

    # Generation indices :
    #       For each track :
    #           - middle of track is > temporal_order
    #           - end if not
    #           - nothing if end < temporal_order
    generation_index = last_indices(tracks_start_end_test, generation_length)

    return piano_train_shared, orchestra_train_shared, np.asarray(train_batches, dtype=np.int32),\
        piano_valid_shared, orchestra_valid_shared, np.asarray(valid_batches, dtype=np.int32),\
        piano_test_shared, orchestra_test_shared, np.asarray(test_batches, dtype=np.int32), np.asarray(generation_index, dtype=np.int32)
