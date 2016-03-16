#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
Repeat
"""
# Numpy
import numpy as np

from Measure.accuracy_measure import accuracy_measure_not_shared
from Data_processing.load_data import load_data_tvt


def train_hopt(temporal_granularity, dataset, max_evals, log_file_path, csv_file_path):
    # Create/reinit log and csv files
    open(csv_file_path, 'w').close()
    open(log_file_path, 'w').close()

    # Init log_file
    with open(log_file_path, 'ab') as log_file:
        log_file.write((u'# Repeat model\n').encode('utf8'))
        log_file.write((u'# Temporal granularity : ' + temporal_granularity + '\n').encode('utf8'))

    # Define hyper-parameter search space
    batch_size = 100  # not an relevant hp for this model, so not included in the search procedure

    orch, orch_mapping, piano, piano_mapping, train_index, val_index, _ \
        = load_data_tvt(data_path=dataset,
                        log_file_path='bullshit.txt',
                        temporal_granularity=temporal_granularity,
                        temporal_order=1,
                        shared_bool=False,
                        minibatch_size=batch_size,
                        split=(0.7, 0.1, 0.2))

    # Model validation ###########
    all_val_idx = []
    for i in xrange(0, len(val_index)):
        all_val_idx.extend(val_index[i])
    all_val_idx = np.array(all_val_idx)     # Oui, c'est dégueulasse, mais vraiment
    true_frame = orch[all_val_idx]
    accuracy = accuracy_measure_not_shared(true_frame[1:], true_frame[0:-1])
    mean_accuracy = 100 * np.mean(accuracy)
    error = -mean_accuracy
    ##############################

    # Log
    with open(log_file_path, 'ab') as log_file:
        log_file.write((u'# Accuracy :  {}'.format(mean_accuracy)).encode('utf8'))
        log_file.write((u'###################\n').encode('utf8'))
    # Print
    print((u'# Accuracy :  {}'.format(mean_accuracy)).encode('utf8'))
    print((u'###################\n').encode('utf8'))

    return error
