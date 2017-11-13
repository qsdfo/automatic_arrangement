#!/usr/bin/env python
# -*- coding: utf8 -*-


from sklearn import datasets, linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

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
	x_ = []
	y_ = []
	for config in configs:
		for fold in range(10):
			path_result_file = os.path.join(folder, config, str(fold), "result.csv")
			# Read result.csv
			with open(path_result_file, "rb") as f:
				reader = csv.DictReader(f, delimiter=';')
				elem = reader.next()
				Xentr = float(elem["loss"])
				acc = float(elem["accuracy"])
				if acc > 3:
					x_.append(Xentr)
					y_.append(acc)

x = np.asarray(x_)
y = np.asarray(y_)

# Linear regression
regr = linear_model.LinearRegression()
regr.fit(np.reshape(x, (-1, 1)), y)
y_reg = regr.predict(np.reshape(x, (-1, 1)))

# Create a trace
mse =  mean_squared_error(y, y_reg)

plt.scatter(x, y, c="b", alpha=0.5)
plt.plot(x, y_reg, c="r")
plt.xlabel("Xentr")
plt.ylabel("Acc")
plt.title("Linear regression between accuracy scores and binary cross-entropy.\n MSE = " + str(mse))

# Plot and embed in ipython notebook!
plt.savefig('Xentr_acc_link.pdf')
