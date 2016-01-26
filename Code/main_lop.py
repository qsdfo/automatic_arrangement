#!/usr/bin/env python
# -*- coding: utf8 -*-

# Main script for LOP
import csv
import os

from Data_processing import load_data

# Select a model (path to the .py file)
# Two things define a model : it's architecture and the time granularity
from Models.Temporal_RBM.temporal_binary_rbm import train
model_path = u'Temporal_RBM'
temporal_granularity = u'frame_level'

# Log file
log_file_path = u'log.txt'

# Build the matrix database (stored in a data.p file in Data) from a music XML database
database = '../Data/data.p'

# Set hyperparameters (can be a grid)
result_folder = u'../Results/' + temporal_granularity + u'/' + model_path + u'/'
result_file = result_folder + u'results.csv'

# Config is set now, no need to modify source below for standard use
############################################################################
############################################################################
############################################################################

# Init log_file
log_file = open(log_file_path, 'w')
log_file.write((u'## LOG FILE : \n').encode('utf8'))
log_file.write((u'## Model : ' + model_path + '\n').encode('utf8'))
log_file.write((u'## Temporal granularity : ' + temporal_granularity + '\n').encode('utf8'))

# Check if the result folder exists
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Import hyperparams from a csv file (config.csv) and run each row in this csv
hyper_parameters = {}
config_file_path = u'/config.csv'
with open(config_file_path, 'rb') as csvfile:
    config_csv = csv.reader(csvfile, delimiter=';')
    headers_config = config_csv.next()
    config_number = 0
    for row in config_csv:
        column = 0
        this_hyperparam = {}
        for hyperparam in headers_config:
            this_hyperparam[hyperparam] = row[column]
            column += 1
        hyper_parameters[config_number] = this_hyperparam
        config_number += 1
config_number_to_train = config_number
# Import from result.csv the alreday tested configurations in a dictionnary
checked_config = {}
with open(result_file, 'rb') as csvfile:
    result_csv = csv.reader(csvfile, delimiter=';')
    headers_result = result_csv.next()
    result_number = 0
    for row in result_csv:
        column = 0
        this_hyperparam = {}
        for hyperparam in headers_config:  # /!\ Note that we use the header of the config file
            this_hyperparam[hyperparam] = row[column]
            column += 1
        checked_config[result_number] = this_hyperparam
        config_number += 1
config_number_trained = config_number
log_file.write((u'## Number of congif to train : ' + config_number_to_train + '\n').encode('utf8'))
log_file.write((u'## Number of congif already trained : ' + config_number_trained + '\n').encode('utf8'))
log_file.write((u'###############################################\n\n').encode('utf8'))

# Compare granularity with granularity in the config_file

# Train the model, looping over the hyperparameters configurations
config_trained = 0
for config_hp in hyper_parameters.itervalues():
    log_file.write((u'## Config ' + config_trained + '\n').encode('utf8'))
    # Before training for an hyperparam point, check if it has already been tested.
    #   If it's the case, values would be stored in an other CSV files (result.csv), with its performance
    NO_RUN = False
    config_number = 0   # Counter on the number of config
    for result_hp in checked_config.itervalues():
        if result_hp == config_hp:
            NO_RUN = True
            break
    if NO_RUN:
        log_file.write(("This config has already been tested\n\n").encode('utf8'))
        break

    log_file.close()
    # Train the model
    trained_model, performance = train(config_hp, database, log_file_path)
    log_file = open(log_file_path, 'ab')
    log_file.write(('Performance : ').encode('utf8'))

    # Write the result in result.csv
