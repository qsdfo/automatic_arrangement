#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import copy


def tvt_minibatch(log_file_path, valid_index, minibatch_size, shuffle_type='all', split=(0.7, 0.1, 0.2)):
    """
    Used to shuffle the dataset at each iteration.

    valid_index:   numpy array
                contains the valid indexes, i.e. indexes which more than temporal_order after the beginning
                of a track
    shuffle_type:   * all = shuffle all db, THEN build tvt indices
                    * block = split between tvt, then shuffle inside t,v, or t
                    * none = no shuffling at all
    split:      split proportion of the whole dataset between train, validate and test datasets
    """
    # Default is none
    shuffle = False
    if shuffle_type == 'all':
        np.random.shuffle(valid_index)
        shuffle = False
    elif shuffle_type == 'block':
        shuffle = True

    n = valid_index.shape[0]

    minibatch_start = 0

    # TRAIN
    minibatches_train = []
    last_train = int(n * split[0])
    n_batch_train = int((n * split[0]) // minibatch_size)
    if shuffle:
        np.random.shuffle(valid_index[0:last_train])
    for i in range(n_batch_train):
        minibatches_train.append(valid_index[minibatch_start:
                                 minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != last_train:
        # Make a minibatch out of what is left
        minibatches_train.append(valid_index[minibatch_start:last_train])
        n_batch_train = n_batch_train + 1

    # VALIDATE
    minibatches_validate = []
    minibatch_start = last_train
    last_validate = int(last_train + n * split[1])
    n_batch_validate = int((n * split[1]) // minibatch_size)
    if shuffle:
        np.random.shuffle(valid_index[minibatch_start:last_validate])
    for i in range(n_batch_validate):
        minibatches_validate.append(valid_index[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != last_validate:
        # Make a minibatch out of what is left
        minibatches_validate.append(valid_index[minibatch_start:last_validate])
        n_batch_validate = n_batch_validate + 1

    # TEST
    minibatches_test = []
    minibatch_start = last_validate
    last_test = int(last_validate + n * split[2])
    n_batch_test = int((n * split[2]) // minibatch_size)
    if shuffle:
        np.random.shuffle(valid_index[minibatch_start:last_test])
    for i in range(n_batch_test):
        minibatches_test.append(valid_index[minibatch_start:
                                minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != last_test:
        # Make a minibatch out of what is left
        minibatches_test.append(valid_index[minibatch_start:last_test])
        n_batch_test = n_batch_test + 1

    return np.array(minibatches_train), np.array(minibatches_validate), np.array(minibatches_test)


def tvt_minibatch_seq(log_file_path, valid_index, temporal_order, shuffle=True, split=(0.7, 0.1, 0.2)):
    """
    Used to shuffle the dataset at each iteration.

    valid_index:   numpy array
                contains the valid indexes, i.e. indexes which more than temporal_order after the beginning
                of a track
    shuffle_type:   * all = shuffle all db, THEN build tvt indices
                    * block = split between tvt, then shuffle inside t,v, or t
                    * none = no shuffling at all
    split:      split proportion of the whole dataset between train, validate and test datasets
    """

    n = valid_index.shape[0]
    # TRAIN
    minibatches_train = []
    n_batch_train = int(n * split[0])
    for i in range(n_batch_train):
        minibatches_train.append(range(valid_index[i] - temporal_order, valid_index[i]))
    if shuffle:
        np.random.shuffle(minibatches_train)

    # VALIDATE
    minibatches_validate = []
    n_batch_validate = int(n * (split[0] + split[1]))
    for i in range(n_batch_train, n_batch_validate):
        minibatches_validate.append(range(valid_index[i] - temporal_order, valid_index[i]))
    if shuffle:
        np.random.shuffle(minibatches_validate)

    # TEST
    minibatches_test = []
    n_batch_test = int(n * (split[0] + split[1] + split[2]))
    for i in range(n_batch_validate, n_batch_test):
        minibatches_test.append(range(valid_index[i] - temporal_order, valid_index[i]))
    if shuffle:
        np.random.shuffle(minibatches_test)

    return np.array(minibatches_train), np.array(minibatches_validate), np.array(minibatches_test)


def k_fold_cross_validation(log_file_path, valid_index, minibatch_size, split=(0.7, 0.1, 0.2)):
    n = valid_index.shape[0]
    k = int(1 / split[2])

    minibatches_train = {}
    minibatches_validate = {}
    minibatches_test = {}

    for step in range(0, k):
        # Get train, validate, test indices
        train, validate, test = tvt_minibatch(log_file_path, copy.copy(valid_index), minibatch_size, shuffle_type='block', split=split)
        minibatches_train[step] = train
        minibatches_validate[step] = validate
        minibatches_test[step] = test
        # permute on the valide indices
        n_end = int(n * split[2])
        valid_index = np.concatenate((valid_index[-n_end:], valid_index[0:-n_end]))
    return minibatches_train, minibatches_validate, minibatches_test

if __name__ == '__main__':
    valid_index = np.arange(20)
    k_fold_cross_validation('test.txt', valid_index, 3)
