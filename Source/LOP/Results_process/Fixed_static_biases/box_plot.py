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
            ("LSTM_plugged_base/2", "std \n wd \n 5e-1"),
            ("LSTM_plugged_base/3", "std \n wd \n 1e-2"),
            ("LSTM_plugged_base/4", "std \n wd \n 5e-2"),
            ("LSTM_plugged_base/5", "std \n wd \n 1e-3"),
#            ("LSTM_static_bias/0", "fixed \n biases"),
#            ("LSTM_static_bias/1", "fb \n wd \n 1e-1"),
#            ("LSTM_static_bias/2", "fb \n wd \n 5e-1"),
#            ("LSTM_static_bias/3", "fb \n wd \n 1e-2"),
#            ("LSTM_static_bias/4", "fb \n wd \n 5e-2"),
#            ("LSTM_static_bias/5", "fb \n wd \n 1e-3"),
    ]
    
    data_Xent = []
    data_acc = []
    epochs = []
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
        epoch = []
        with open(os.path.join(root, path, 'best_epoch.txt'), 'rb') as ff:
            for line in ff:
                epoch.append(int(line.rstrip('\n')))
        
        epochs.append(epoch)
        data_Xent.append(this_data_Xent)
        data_acc.append(this_data_acc)
        labels.append(title)
    
    # Sort by keys values
#    plt.boxplot(data_Xent, labels=labels)
#    plt.ylabel('Xent')
#    plt.xlabel('Model')
#    plt.title('10-folds boxplots of the binary cross-entropy \n with and without pre-computed static biases')
#    plt.tight_layout()
#    plt.show("Xent.pdf")

#    plt.clf()
#    plt.ylabel('accuracy (%)')
#    plt.xlabel('Model')
#    plt.title('10-folds boxplots of the accuracy measure \n with and without pre-computed static biases')
#    plt.boxplot(data_acc, labels=labels)
#    plt.tight_layout()
#    plt.show("accuracy.pdf")
#    
#    plt.clf()
#    plt.ylabel('Number of epochs')
#    plt.xlabel('Model')
#    plt.title('10-folds boxplots of the number of training epoch \n with and without pre-computed static biases')
#    plt.boxplot(epochs, labels=labels)
#    plt.tight_layout()
#    plt.show()
    
    plt.clf()
    # Scatter plot epoch/acc
    plt.scatter(epochs, data_acc, c="b", alpha=0.5)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Relation between epoch and accuracy \n for pre-computed biases models")
    plt.xlim(0, 105)
    plt.ylim(35, 47)
    plt.show()