#!/usr/bin/env python
# -*- coding: utf8 -*-

# Main script for LOP
import os

import theano
theano.config.compute_test_value = 'warn'

# Select a model (path to the .py file)
# Two things define a model : it's architecture and the time granularity
from Models.Temporal_RBM.train_hopt import train_hopt
model_name = u'Temporal_RBM'
temporal_granularity = u'frame_level'

# Log file
MAIN_DIR = os.getcwd().decode('utf8') + u'/'
log_file_path = MAIN_DIR + u'log'

# Build the matrix database (stored in a data.p file in Data) from a music XML database
dataset = MAIN_DIR + u'../Data/data.p'

# Set hyperparameters (can be a grid)
result_file = u'../Results/' + temporal_granularity +\
              '/' + model_name + u'/hopt_results.csv'

# Config is set now, no need to modify source below for standard use
############################################################################
############################################################################


########################################################################
# Train step
########################################################################
max_evals = 10  # number of hyper-parameter configurations evaluated
best = train_hopt(temporal_granularity, dataset, max_evals, log_file_path, result_file)
print best
