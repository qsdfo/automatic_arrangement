#!/usr/bin/env python
# -*- coding: utf8 -*-

import glob
import os
import shutil
import cPickle as pkl
import time
# Load data
from load_data import load_data_test
from generate import generate
import logging

def generate_midi(path, config=None, generation_length=50, seed_size=10, quantization_write=None):
    # Generate midi files for a given path, which correspond to a set of training parameters :
    # model, optimisation, granularity, conitous/discrete, quantization
    # If config is None, process all configurations

    # log file
    ############################################################
    # Logging
    ############################################################
    log_file_path = 'generate_log'
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_file_path,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logger_generate = logging.getLogger('generate')

    ############################################################
    # Grab config files
    ############################################################
    if config:
        config_list = [config]
    else:
        config_list = glob.glob(path + '/*')

    for config_folder in config_list:
        ############################################################
        # Load the model and config
        ############################################################
        model_path = config_folder + '/model.pkl'
        model = pkl.load(open(model_path, 'rb'))
        param_path = config_folder + 'config.pkl'
        space = pkl.load(open(param_path, 'rb'))
        model_param = space['model']
        script_param = space['script']
        metadata_path = script_param['data'] + '/metadata.pkl'

        if quantization_write is None:
            quantization_write = script_param['quantization']

        ############################################################
        # Create generation folder
        ############################################################
        generated_folder = config_folder + '/generated_sequences'
        if not os.path.isdir(generated_folder):
            os.makedirs(generated_folder)

        ############################################################
        # Load data
        ############################################################
        time_load_0 = time.time()
        piano_test, orchestra_test, _, generation_index \
            = load_data_test(script_param['data_folder'],
                             model_param['temporal_order'],
                             model_param['batch_size'],
                             binary_unit=script_param['binary_unit'],
                             skip_sample=script_param['skip_sample'],
                             logger_load=logger_generate)
        time_load_1 = time.time()
        logger_generate.info('TTT : Loading data took {} seconds'.format(time_load_1-time_load_0))

        ############################################################
        # Generate
        ############################################################
        time_generate_0 = time.time()
        generate(model,
                 piano_test, orchestra_test, generation_index, metadata_path,
                 generation_length, seed_size, quantization_write,
                 generated_folder, logger_generate)
        time_generate_1 = time.time()
        logger_generate.info('TTT : Generating data took {} seconds'.format(time_generate_1-time_generate_0))

def plot_weights(path, config=None):
    

# def process_result_csv(path):
#
# def statistics():
#     # Compute statistics such as bar plots for haparams

def clean(path):
    list_dir = glob.glob(path + '/*')
    for dirname in list_dir:
        list_file = os.listdir(dirname)
        NO_RESULT_FILE = 'result.csv' not in list_file
        NO_CONFIG_FILE = 'config.pkl' not in list_file
        NO_MODEL_FILE = 'model.pkl' not in list_file
        if NO_CONFIG_FILE or NO_RESULT_FILE or NO_MODEL_FILE:
            shutil.rmtree(dirname)

if __name__ == '__main__':
    clean('/home/aciditeam-leo/Aciditeam/lop/Results/event_level/discrete_units/quantization_4/gradient_descent/LSTM')
