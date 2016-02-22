#!/usr/bin/env python
# -*- coding: utf8 -*-

# Main script for LOP
import csv
import os

# Select a model (path to the .py file)
# Two things define a model : it's architecture and the time granularity
from Models.Temporal_RBM.temporal_binary_rbm import train, save
model_name = u'Temporal_RBM'
temporal_granularity = u'frame_level'

# Log file
MAIN_DIR = os.getcwd().decode('utf8') + u'/'
log_file_path = MAIN_DIR + u'log'

# Build the matrix database (stored in a data.p file in Data) from a music XML database
database = MAIN_DIR + u'../Data/data.p'

# Set hyperparameters (can be a grid)
result_folder = MAIN_DIR + u'../Results/' + temporal_granularity + u'/' + model_name + u'/'
result_file = result_folder + u'results.csv'

# Config is set now, no need to modify source below for standard use
############################################################################
############################################################################
############################################################################

# Init log_file
log_file = open(log_file_path, 'wb')
log_file.write((u'## LOG FILE : \n').encode('utf8'))
log_file.write((u'## Model : ' + model_name + '\n').encode('utf8'))
log_file.write((u'## Temporal granularity : ' + temporal_granularity + '\n').encode('utf8'))

# Check if the result folder exists
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Import hyperparams from a csv file (config.csv) and run each row in this csv
hyper_parameters = {}
config_file_path = u'config.csv'
with open(config_file_path, 'rb') as csvfile:
    config_csv = csv.DictReader(csvfile, delimiter=',')
    headers_config = config_csv.fieldnames
    config_number = 0
    for row in config_csv:
        hyper_parameters[config_number] = row
        config_number += 1
config_number_to_train = config_number
# Import from result.csv the alreday tested configurations in a dictionnary
checked_config = {}
if os.stat(result_file).st_size == 0:
    # Empty file
    config_number_trained = 0
else:
    with open(result_file, 'rb') as csvfile2:
        result_csv = csv.DictReader(csvfile2, delimiter=',')
        headers_result = result_csv.fieldnames
        result_number = 0
        for row in result_csv:
            # Extract sub-dictionary from the result_dictionary
            checked_config[result_number] = dict([(i, row[i]) for i in headers_config if i in row])
            result_number += 1
    config_number_trained = result_number
log_file.write((u'## Number of config to train : %d \n' % config_number_to_train).encode('utf8'))
log_file.write((u'## Number of config already trained : %d \n' % config_number_trained).encode('utf8'))
log_file.write((u'\n###############################################\n\n').encode('utf8'))

# Train the model, looping over the hyperparameters configurations
config_train = 0
for config_hp in hyper_parameters.itervalues():
    log_file.write((u'\n###############################################\n').encode('utf8'))
    log_file.write((u'## Config ' + str(config_train) + '\n').encode('utf8'))
    # Check the temporal granularity
    if not temporal_granularity == config_hp['temporal_granularity']:
        log_file.write("The temporal granularity in the folder name is not the same as the one announced in the config file\n").encode('utf8')
        config_train += 1
        continue
    # Before training for an hyperparam point, check if it has already been tested.
    #   If it's the case, values would be stored in an other CSV files (result.csv), with its performance
    NO_RUN = False
    for result_hp in checked_config.itervalues():
        if result_hp == config_hp:
            NO_RUN = True
            break
    if NO_RUN:
        log_file.write(("This config has already been tested\n").encode('utf8'))
        config_train += 1
        continue

    log_file.close()
    # Train the model
    trained_model, precision, recall, accuracy = train(config_hp, database, log_file_path)

    # Write logs
    log_file = open(log_file_path, 'ab')
    log_file.write(('## Performance : \n').encode('utf8'))
    log_file.write(('    Precision = {}'.format(precision)).encode('utf8'))
    log_file.write(('    Recall = {}'.format(recall)).encode('utf8'))
    log_file.write(('    Accuracy = {}'.format(accuracy)).encode('utf8'))

    # Store results in the configuration dictionary
    config_hp['precision'] = 100 * precision
    config_hp['recall'] = 100 * recall
    config_hp['accuracy'] = 100 * accuracy

    # Keep count of the number of config trained
    config_index = config_number_trained + config_train  # Index of the config
    config_train += 1

    # Index config
    config_hp['index'] = config_index
    # Store the net in a csv file
    save_net_path = result_folder + str(config_index)
    # Save the structure in a folder (csv files)
    save(trained_model, save_net_path)

    # Write the result in result.csv
    with open(result_file, 'ab') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=headers_result)
        count = 0
        writer.writerow(config_hp)

log_file.close()
