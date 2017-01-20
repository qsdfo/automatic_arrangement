#!/usr/bin/env python
# -*- coding: utf8 -*-

# Generer exemples corrompus
# Calculer l'accuracy corrompue

import numpy as np
import os
import cPickle as pkl
from load_data import load_data_valid
from results_folder_generate import generate_midi
from acidano.utils.init import shared_zeros
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('generate')

def generate_corrupted_results(config_folder, data_folder, generation_length, seed_size):
    # Generation
    # generate_midi(config_folder, data_folder, generation_length, seed_size, 4, None, None)
    generate_midi(config_folder, data_folder, generation_length, seed_size, 4, 'piano', None)
    generate_midi(config_folder, data_folder, generation_length, seed_size, 4, 'orchestra', None)
    # generate_midi(config_folder, data_folder, generation_length, seed_size, 4, 'orchestra_and_piano', None)

    # Create result folder
    corrupted_folder = config_folder + '/corrupted_accuracy/'
    if not os.path.isdir(corrupted_folder):
        os.mkdir(corrupted_folder)

    # Load parameters
    params = pkl.load(open(config_folder + '/config.pkl', "rb"))
    script_param = params['script']
    model_param = params['model']

    # Load model
    model_path = config_folder + '/model.pkl'
    model = pkl.load(open(model_path, 'rb'))

    # Load data
    piano_valid, orchestra_valid, valid_index \
        = load_data_valid(script_param['data_folder'],
                          model.checksum_database['piano_valid'], model.checksum_database['orchestra_valid'],
                          model_param['temporal_order'],
                          model_param['batch_size'],
                          binary_unit=script_param['binary_unit'],
                          skip_sample=script_param['skip_sample'],
                          logger_load=logger)

    n_val_batches = len(valid_index)

    # Accuracy normal (just to compare)
    validation_error = model.get_validation_error(piano_valid, orchestra_valid, name='validation_error_corrupted_piano')
    accuracy = []
    for batch_index in xrange(n_val_batches):
        _, _, accuracy_batch = validation_error(valid_index[batch_index])
        accuracy += [accuracy_batch]
    mean_accuracy = 100 * np.mean(accuracy)

    with open(corrupted_folder + '/normal.txt', 'wb') as f:
        f.write(str(mean_accuracy))

    # Accuracy corrupted piano
    piano_valid_corrupted = shared_zeros(piano_valid.get_value(borrow=True).shape)
    validation_error = model.get_validation_error(piano_valid_corrupted, orchestra_valid, name='validation_error_corrupted_piano')
    accuracy = []
    for batch_index in xrange(n_val_batches):
        _, _, accuracy_batch = validation_error(valid_index[batch_index])
        accuracy += [accuracy_batch]
    mean_accuracy_corrupted_piano = 100 * np.mean(accuracy)

    with open(corrupted_folder + '/corrupted_piano.txt', 'wb') as f:
        f.write(str(mean_accuracy_corrupted_piano))

    # Accuracy corrupted piano
    orchestra_valid_corrupted = shared_zeros(orchestra_valid.get_value(borrow=True).shape)
    validation_error = model.get_validation_error(piano_valid, orchestra_valid_corrupted, name='validation_error_corrupted_orchestra')
    accuracy = []
    for batch_index in xrange(n_val_batches):
        _, _, accuracy_batch = validation_error(valid_index[batch_index])
        accuracy += [accuracy_batch]
    mean_accuracy_corrupted_orchestra = 100 * np.mean(accuracy)

    with open(corrupted_folder + '/corrupted_orchestra.txt', 'wb') as f:
        f.write(str(mean_accuracy_corrupted_orchestra))

if __name__ == '__main__':
    config_folder = '/home/aciditeam-leo/Aciditeam/lop/Results/event_level/discrete_units/quantization_4/gradient_descent/cRnnRbm/8664641'
    data_folder = '/home/aciditeam-leo/Aciditeam/lop/Data'
    generate_corrupted_results(config_folder, data_folder, generation_length=100, seed_size=20)
