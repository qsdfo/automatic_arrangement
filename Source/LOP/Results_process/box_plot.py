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

for folder in folders:
	configs = os.listdir(folder)
	loss_k_folds = {}
	acc_k_folds = {}
	for config in configs:
		for fold in range(10):
			path_result_file = os.path.join(folder, config, str(fold), "result.csv")
			# Read result.csv
			with open(path_result_file, "rb") as f:
				reader = csv.DictReader(f, delimiter=';')
				elem = reader.next()
				if fold in loss_k_folds.keys():
					loss_k_folds[fold].append(elem["loss"])
					acc_k_folds[fold].append(elem["accuracy"])
				else:
					loss_k_folds[fold] = [elem["loss"]]
					acc_k_folds[fold] = [elem["accuracy"]]


def plot_dict(dict, title, xaxis, yaxis, filename):
	data = []

	for k, v in dict.iteritems():

		trace = go.Box(
			y=v,
			name="Fold_" + str(k),
		)

		data.append(trace)

	layout = {
		'title': title,
	    'xaxis': {
	        'title': xaxis,
	    },
	    'yaxis': {
	        'title': yaxis,
	    },
	    'boxmode': 'group',
	}

	fig = go.Figure(data=data, layout=layout)

	# Plot data
	py.plot(fig, filename=filename)

plot_dict(loss_k_folds, "Difference of performances between folds (LSTM_based). Neg-ll", "fold index", "neg-ll", "k_folds_loss")
plot_dict(acc_k_folds, "Difference of performances between folds (LSTM_based). Accuracy", "fold index", "neg-ll", "k_folds_acc")