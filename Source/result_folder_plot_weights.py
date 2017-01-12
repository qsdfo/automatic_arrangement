#!/usr/bin/env python
# -*- coding: utf8 -*-


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
    model.save_weights(plot_folder)
