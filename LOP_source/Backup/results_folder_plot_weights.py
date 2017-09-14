#!/usr/bin/env python
# -*- coding: utf8 -*-

import time
import cPickle as pkl
import os

def plot_weights(path, logger_plot):
    # Plot weights for a model contained in a configuration path
    logger_plot.info('Plotting weights for ' + path)
    ############################################################
    # Load the model and config
    ############################################################
    model_path = path + '/model.pkl'
    model = pkl.load(open(model_path, 'rb'))

    ############################################################
    # Create plot folder
    ############################################################
    plot_folder = path + '/plot_weights'
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)

    ############################################################
    # Plot weights
    ############################################################
    time_load_0 = time.time()
    model.save_weights(plot_folder)
    time_load_1 = time.time()
    logger_plot.info('TTT : Plotting took {} seconds'.format(time_load_1-time_load_0))
