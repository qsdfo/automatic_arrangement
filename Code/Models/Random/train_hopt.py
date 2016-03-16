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
from hyperopt import hp, fmin, tpe
# CSV
import csv

from Measure.accuracy_measure import accuracy_measure_not_shared
from Data_processing.load_data import load_data_tvt


def train_hopt(temporal_granularity, dataset, max_evals, log_file_path, csv_file_path):
    # Create/reinit log and csv files
    open(csv_file_path, 'w').close()
    open(log_file_path, 'w').close()

    # Init log_file
    with open(log_file_path, 'ab') as log_file:
        log_file.write((u'# Random model\n').encode('utf8'))
        log_file.write((u'# Temporal granularity : ' + temporal_granularity + '\n').encode('utf8'))

    # Define hyper-parameter search space
    header = ['bernoulli_param', 'accuracy']
    space = (hp.uniform('bernoulli_param', 0, 1))
    batch_size = 100  # not an relevant hp for this model, so not included in the search procedure

    global run_counter
    run_counter = 0

    def run_wrapper(params):
        global run_counter
        run_counter += 1

        bernoulli_param = params

        # log
        with open(log_file_path, 'ab') as log_file:
            log_file.write((u'\n###################').encode('utf8'))
            log_file.write((u'# Config :  {}'.format(run_counter)).encode('utf8'))
            log_file.write((u'# bernoulli_param :  {}'.format(bernoulli_param)).encode('utf8'))
        # print
        print((u'\n###################').encode('utf8'))
        print((u'# Config :  {}'.format(run_counter)).encode('utf8'))
        print((u'# bernoulli_param :  {}'.format(bernoulli_param)).encode('utf8'))

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
        pred_frame = bernoulli.rvs(bernoulli_param, size=true_frame.shape)
        accuracy = accuracy_measure_not_shared(true_frame, pred_frame)
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

        # Write the result in result.csv
        with open(csv_file_path, 'ab') as csvfile:
            bernoulli_param = params
            writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=header)
            dico_res = {'bernoulli_param': bernoulli_param,
                        'accuracy': mean_accuracy}
            writer.writerow(dico_res)

        return error

    with open(csv_file_path, 'ab') as csvfile:
        # Write headers if they don't already exist
        writerHead = csv.writer(csvfile, delimiter=',')
        writerHead.writerow(header)

    best = fmin(run_wrapper, space, algo=tpe.suggest, max_evals=max_evals)

    return best
