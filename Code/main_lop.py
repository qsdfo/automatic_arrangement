#!/usr/bin/env python
# -*- coding: utf8 -*-

# Main script for LOP
import unicodecsv as csv
import os
import numpy as np


def average_dict(dico):
    mean_dic = 0
    counter = 0
    for v in dico.itervalues():
        counter += 1
        mean_dic += np.mean(v)
    return mean_dic / counter


# Select a model (path to the .py file)
# Two things define a model : it's architecture and the time granularity
from Models.Temporal_RBM.temporal_binary_rbm import train, save
model_name = u'Temporal_RBM'
temporal_granularity = u'full_event_level'

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

########################################################################
# Hyper parameters
########################################################################
# Import hyperparams from a csv file (config.csv) and run each row in this csv
hyper_parameters = {}
config_file_path = MAIN_DIR + u'Models/' + model_name + u'/config.csv'
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
headers_result = [u'index'] + headers_config + [u'precision', u'recall', u'accuracy']
config_number_trained = 0
RESULT_FILE_ALREADY_EXISTS = False
if os.path.isfile(result_file) and (os.stat(result_file).st_size > 0):
    # File exists and is not empty
    RESULT_FILE_ALREADY_EXISTS = True
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


########################################################################
# Train & evaluate
########################################################################
# Train the model, looping over the hyperparameters configurations
config_train = 0
for config_hp in hyper_parameters.itervalues():
    log_file.write((u'\n###############################################\n').encode('utf8'))
    log_file.write((u'## Config ' + str(config_train) + '\n').encode('utf8'))
    print((u'\n###############################################\n').encode('utf8'))
    print((u'## Config ' + str(config_train) + '\n').encode('utf8'))

    # Check the temporal granularity
    if not temporal_granularity == config_hp[u'temporal_granularity']:
        log_file.write(u"The temporal granularity in the folder name is not the same as the one announced in the config file\n".encode('utf8'))
        print(u"The temporal granularity in the folder name is not the same as the one announced in the config file\n".encode('utf8'))
        continue

    # Before training for an hyperparam point, check if it has already been tested.
    #   If it's the case, values would be stored in an other CSV files (result.csv), with its performance
    NO_RUN = False
    for result_hp in checked_config.itervalues():
        if result_hp == config_hp:
            NO_RUN = True
            break
    if NO_RUN:
        log_file.write((u"This config has already been tested\n").encode('utf8'))
        print((u"This config has already been tested\n").encode('utf8'))
        continue

    log_file.close()
    # Train the model
    trained_model, record = train(config_hp, database, log_file_path)
    ##########
    # This is extremly important to keep in mind that when using
    # k-fold cross-validation, only the last 10th network is returned, hence
    # it can't be used as the "best" network.
    # The best way to get a generative network at the end of the training process
    # is to get the set of hyper-parameters that obtained the best mean performance
    # over the k fold and train it on the whole database this time (no valid, no test)
    ##########
    ##########

    precision = record['precision']
    recall = record['recall']
    accuracy = record['accuracy']

    # Write logs
    log_file = open(log_file_path, 'ab')
    log_file.write((u'\n## Performance : \n').encode('utf8'))
    log_file.write((u'    Precision = {}\n'.format(average_dict(precision))).encode('utf8'))
    log_file.write((u'    Recall = {}\n'.format(average_dict(recall))).encode('utf8'))
    log_file.write((u'    Accuracy = {}\n\n'.format(average_dict(accuracy))).encode('utf8'))

    # Store results in the configuration dictionary
    config_hp[u'precision'] = 100 * average_dict(precision)
    config_hp[u'recall'] = 100 * average_dict(recall)
    config_hp[u'accuracy'] = 100 * average_dict(accuracy)

    # Keep count of the number of config trained
    config_index = config_number_trained + config_train  # Index of the config
    config_train += 1

    # Index config
    config_hp[u'index'] = config_index
    # Store the net in a csv file
    save_net_path = result_folder + unicode(str(config_index)) + u'/'
    # Save the structure in a folder (csv files)
    save(trained_model, save_net_path)

    if not RESULT_FILE_ALREADY_EXISTS:
        with open(result_file, 'ab') as csvfile:
            # Write headers if they don't already exist
            writerHead = csv.writer(csvfile, delimiter=',')
            writerHead.writerow(headers_result)
            RESULT_FILE_ALREADY_EXISTS = True

    # Write the result in result.csv
    with open(result_file, 'ab') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=headers_result)
        count = 0
        writer.writerow(config_hp)

log_file.close()
