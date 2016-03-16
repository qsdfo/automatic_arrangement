#!/usr/bin/env python
# -*- coding: utf8 -*-

# Main script for LOP
import os

# Select a model (path to the .py file)
# Two things define a model : it's architecture and the time granularity
from Models.CRBM.train_hopt import train_hopt
model_name = u'CRBM'
temporal_granularity = u'frame_level'

# Get main dir
MAIN_DIR = os.getcwd().decode('utf8') + u'/'

# Build the matrix database (stored in a data.p file in Data) from a music XML database
dataset = MAIN_DIR + u'../Data/data.p'

# Set hyperparameters (can be a grid)
result_folder = MAIN_DIR + u'../Results/' + temporal_granularity + '/' + model_name
result_file = result_folder + u'/hopt_results.csv'
log_file_path = result_folder + u'log'

# Config is set now, no need to modify source below for standard use
############################################################################
############################################################################

max_evals = 50  # number of hyper-parameter configurations evaluated

# Check is the result folder exists
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

best = train_hopt(temporal_granularity, dataset, max_evals, log_file_path, result_file)
print best
