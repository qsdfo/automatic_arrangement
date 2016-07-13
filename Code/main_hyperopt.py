#!/usr/bin/env python
# -*- coding: utf8 -*-

# Main script for LOP
import os
import csv
from hyperopt import fmin, tpe

# Reminder for plotting tools
import matplotlib.pyplot as plt
# Histogram
# n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# plt.show()

# Debugging compiler flags
import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'warn'

from load_data import load_data_seq_tvt

# Select a model (path to the .py file)
# Two things define a model : it's architecture and the time granularity
from acidano.models.lop.Variational_LSTM.train import get_header, get_hp_space, train
model_name = u'Variational_LSTM'
temporal_granularity = u'frame_level'

# Get main dir
MAIN_DIR = os.getcwd().decode('utf8') + u'/'

# Build the matrix database (stored in a data.p file in Data) from a music XML database
dataset_path = MAIN_DIR + u'../Data/data.p'

# Set hyperparameters (can be a grid)
result_folder = MAIN_DIR + u'../Results/' + temporal_granularity + '/' + model_name
result_file = result_folder + u'/hopt_results.csv'
log_file_path = MAIN_DIR + u'/' + model_name + u'log'

# Config is set now, no need to modify source below for standard use
############################################################################
############################################################################


def train_hopt(temporal_granularity, dataset_path, max_evals, log_file_path, csv_file_path):
    # Create/reinit log and csv files
    open(csv_file_path, 'w').close()
    open(log_file_path, 'w').close()

    # Init log_file
    with open(log_file_path, 'ab') as log_file:
        log_file.write((u'####################\n').encode('utf8'))
        log_file.write((u'##   L O P\n\n').encode('utf8'))
        log_file.write((u'** ' + model_name + ' **\n\n').encode('utf8'))
        log_file.write((u'# Hyperoptimization\n').encode('utf8'))
        log_file.write((u'# Temporal granularity : ' + temporal_granularity + '\n').encode('utf8'))
    print((u'####################').encode('utf8'))
    print((u'##   L O P\n').encode('utf8'))
    print((u'** ' + model_name + ' **\n').encode('utf8'))
    print((u'# Hyperoptimization').encode('utf8'))
    print((u'# Temporal granularity : ' + temporal_granularity + '').encode('utf8'))
    ########################
    # Define hyper-parameter search space
    header = get_header()
    space = get_hp_space()
    # /!\ temporal_order has to be the first
    assert (header[0] == 'temporal_order'), \
        "temporal_order is not the first hyper-parameter dimension !!"

    global run_counter
    run_counter = 0

    def run_wrapper(params):
        global run_counter
        run_counter += 1
        # log
        with open(log_file_path, 'ab') as log_file:
            log_file.write((u'\n###################\n').encode('utf8'))
            log_file.write((u'# Config :  {}\n'.format(run_counter)).encode('utf8'))
        # print
        print((u'\n###################').encode('utf8'))
        print((u'# Config :  {}'.format(run_counter)).encode('utf8'))

        # Train ##############
        # The temporal_order is needed to build the dataset
        temporal_order = int(params[0])
        dataset = load_data_seq_tvt(data_path=dataset_path,
                                    log_file_path='bullshit.txt',
                                    temporal_granularity=temporal_granularity,
                                    temporal_order=temporal_order,
                                    shared_bool=True,
                                    bin_unit_bool=True,
                                    split=(0.7, 0.1, 0.2))
        accuracy, dico_res = train(params, dataset, temporal_granularity, log_file_path)
        error = -accuracy  # Search for a min
        ######################

        # log
        with open(log_file_path, 'ab') as log_file:
            log_file.write((u'# Accuracy :  {}\n'.format(accuracy)).encode('utf8'))
            log_file.write((u'###################\n\n').encode('utf8'))
        # print
        print((u'# Accuracy :  {}'.format(accuracy)).encode('utf8'))
        print((u'###################\n').encode('utf8'))

        # Write the result in result.csv
        with open(csv_file_path, 'ab') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=header)
            writer.writerow(dico_res)

        return error

    with open(csv_file_path, 'ab') as csvfile:
        # Write headers if they don't already exist
        writerHead = csv.writer(csvfile, delimiter=',')
        writerHead.writerow(header)

    best = fmin(run_wrapper, space, algo=tpe.suggest, max_evals=max_evals)

    return best


if __name__ == "__main__":
    max_evals = 3  # number of hyper-parameter configurations evaluated

    # Check is the result folder exists
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    best = train_hopt(temporal_granularity, dataset_path, max_evals, log_file_path, result_file)
    print best
