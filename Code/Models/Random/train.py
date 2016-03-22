#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
Random generation
"""
# Numpy
import numpy as np
# Scipy Bernoulli
from scipy.stats import bernoulli
# Hyperopt
from hyperopt import hp

from Measure.accuracy_measure import accuracy_measure_not_shared
from Data_processing.load_data import load_data_tvt


# Define hyper-parameter search space
def get_header():
    return ['bernoulli_param', 'accuracy']


def get_hp_space():
    space = (hp.uniform('bernoulli_param', 0, 1),
             hp.quniform('batch_size', 10, 500, 10))
    return space


def train(params, dataset, temporal_granularity, log_file_path):

    bernoulli_param, batch_size = params
    batch_size = int(batch_size)

    # Log them
    with open(log_file_path, 'ab') as log_file:
        log_file.write((u'# bernoulli_param :  {}\n'.format(bernoulli_param)).encode('utf8'))
        log_file.write((u'# batch_size :  {}\n'.format(batch_size)).encode('utf8'))
    # Print
    print((u'# bernoulli_param :  {}'.format(bernoulli_param)).encode('utf8'))
    print((u'# batch_size :  {}'.format(batch_size)).encode('utf8'))

    orch, orch_mapping, piano, piano_mapping, train_index, val_index, _ \
        = load_data_tvt(data_path=dataset,
                        log_file_path='bullshit.txt',
                        temporal_granularity=temporal_granularity,
                        temporal_order=1,
                        shared_bool=False,
                        bin_unit_bool=True,
                        minibatch_size=batch_size,
                        split=(0.7, 0.1, 0.2))

    # Model validation ###########
    all_val_idx = []
    for i in xrange(0, len(val_index)):
        all_val_idx.extend(val_index[i])
    all_val_idx = np.array(all_val_idx)     # Oui, c'est dégueulasse, mais vraiment
    true_frame = orch[all_val_idx]
    pred_frame = bernoulli.rvs(bernoulli_param, size=true_frame.shape)
    accuracy = accuracy_measure_not_shared(true_frame, pred_frame)
    best_accuracy = 100 * np.mean(accuracy)
    dico_res = {'bernoulli_param': bernoulli_param,
                'accuracy': best_accuracy}

    return best_accuracy, dico_res
    ##############################
