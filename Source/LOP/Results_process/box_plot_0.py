#!/usr/bin/env python
# -*- coding: utf8 -*-

import plotly.plotly as py
import plotly.graph_objs as go

import os
import csv
import numpy as np

import LOP.Scripts.config as config


# Collect the data
folders = [
	"k_folds",
]
folders = [config.result_root() + '/' + e for e in folders]

num_params = []
score = []
# Use means over k-folds
for folder in folders:
	configs = os.listdir(folder)
	score_config = []
	for config in configs:
		for fold in range(10):
			with open(path_result_file, "rb") as f:
				reader = csv.DictReader(f, delimiter=';')
				elem = reader.next()
				score_config.append(elem["accuracy"])
	score.append(np.mean(score_config))




N = 1000
random_x = np.random.randn(N)
random_y = np.random.randn(N)

# Create a trace
trace = go.Scatter(
    x = random_x,
    y = random_y,
    mode = 'markers'
)

data = [trace]

# Plot and embed in ipython notebook!
py.plot(data, filename='basic-scatter')

# or plot with: plot_url = py.plot(data, filename='basic-line')