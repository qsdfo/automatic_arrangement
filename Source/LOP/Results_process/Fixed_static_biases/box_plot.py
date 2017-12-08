#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:34:41 2017

@author: leo
"""

# Boxplot with/without fixed static biases

from matplotlib import pyplot as plt
import os
import csv



if __name__ == '__main__':
    # Collect the data
    root = "/Users/leo/Recherche/GitHub_Aciditeam/automatic_arrangement/Experiments/Fixed_static_biases/precomputed_fixed_static_biases_quanti"
    configs = [
            ("LSTM_plugged_base/0", "standard"),
            ("LSTM_plugged_base/1", "std \n weight \n decay \n 1e-1"),
            ("LSTM_plugged_base/2", "std \n wd \n 1e-2"),
            ("LSTM_plugged_base/3", "std \n wd \n 1e-3"),
            ("LSTM_static_bias/0", "fixed \n biases"),
            ("LSTM_static_bias/1", "fb \n wd \n 1e-1"),
            ("LSTM_static_bias/2", "fb \n wd \n 1e-2"),
            ("LSTM_static_bias/3", "fb \n wd \n 1e-3"),
    ]
    
    data_Xent = []
    data_acc = []
    labels = []
    for path, title in configs:
        this_data_Xent = []
        this_data_acc = []
        for fold in range(10):
            path_result_file = os.path.join(root, path, str(fold), "result.csv")
            # Read result.csv
            with open(path_result_file, "rb") as f:
                reader = csv.DictReader(f, delimiter=';')
                elem = reader.next()
                this_Xent = float(elem["Xent"])
                this_acc = float(elem["accuracy"])
                this_data_Xent.append(this_Xent)
                this_data_acc.append(this_acc)
        data_Xent.append(this_data_Xent)
        data_acc.append(this_data_acc)
        labels.append(title)
    
    # Sort by keys values
    plt.boxplot(data_Xent, labels=labels)
    plt.savefig("Xent.pdf")
    plt.clf()
    plt.boxplot(data_acc, labels=labels)
    plt.savefig("accuracy.pdf")