#!/usr/bin/env python
# -*- coding: utf8 -*-


import glob
import logging
import cPickle as pkl
import os

def plot_weights(path, config=None):
    # Plot weights for a given path, which correspond to a set of training parameters :
    # model, optimisation, granularity, conitous/discrete, quantization
    # If config is None, process all configurations

    ############################################################
    # Logging
    ############################################################
    log_file_path = 'weights_log'
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
    logger_plot = logging.getLogger('plot')

    ############################################################
    # Grab config files
    ############################################################
    if config:
        config_list = [config]
    else:
        config_list = glob.glob(path + '/*')

    for config_folder in config_list:
        logger_plot.info('Plotting weights for ' + config_folder)
        ############################################################
        # Load the model and config
        ############################################################
        model_path = config_folder + '/model.pkl'
        model = pkl.load(open(model_path, 'rb'))

        ############################################################
        # Create plot folder
        ############################################################
        plot_folder = config_folder + '/plot_weights'
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)

        ############################################################
        # Plot weights
        ############################################################
        model.save_weights(plot_folder)
