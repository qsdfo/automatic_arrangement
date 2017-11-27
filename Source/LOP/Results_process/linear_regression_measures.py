#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:52:16 2017
Plot relation between two measures and their linear regression.

@author: leo
"""

#!/usr/bin/env python
# -*- coding: utf8 -*-


from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

import os
import shutil
import numpy as np

def compare(x, name_measure_A, y, name_measure_B, result_folder):
    """Compute the linear regression, scatter plot the points in a plan along with the curve derived from the linear reg.
    
    Parameters
    ----------
    x : 1D npy array
        values corresponding to the first measure
    y : 1D npy array
        values corresponding to the second measure
    
    """
    plt.clf()
    N_sample = 1000
    x = x[:N_sample]
    y = y[:N_sample]
    # Linear regression
    regr = linear_model.LinearRegression()
    regr.fit(np.reshape(x, (-1, 1)), y)
    y_reg = regr.predict(np.reshape(x, (-1, 1)))
    
    # Create a trace
    mse =  mean_squared_error(y, y_reg)
    
    plt.scatter(x, y, c="b", alpha=0.5)
    plt.plot(x, y_reg, c="r")
    plt.xlabel(name_measure_A)
    plt.ylabel(name_measure_B)
    plt.title("Linear regression between " + name_measure_A + " and " + name_measure_B + ".\n MSE = " + str(mse))
    
    # Save
    plt.savefig(result_folder + '/' + name_measure_A + '_' + name_measure_B + '_compare.pdf')
    return

def process_measure(config_folders, name_measure_A, name_measure_B, result_folder):
    """Gather the data from different configuration folders
    
    Paramaters
    ----------
    config_folders : str list
        list of strings containing the configurations from which we want to collect results
    name_measure_A : str
        name of the first measure (has to match a .npy file in the corresponding configuration folder)
    name_measure_B : str
        name of the second measure
        
    """
    measure_A_list = []
    measure_B_list = []
    for config_folder in config_folders:
        # Read npy
        measure_A_list.append(np.load(os.path.join(config_folder, name_measure_A + '.npy')))
        measure_B_list.append(np.load(os.path.join(config_folder, name_measure_B + '.npy')))
    measure_A = np.concatenate(measure_A_list)
    measure_B = np.concatenate(measure_B_list)
    compare(measure_A, name_measure_A, measure_B, name_measure_B, result_folder)
    return

if __name__ == '__main__':
    source_folder = '/Users/leo/Recherche/GitHub_Aciditeam/lop/Results/MEASURE/scatter_measures_link'
    config_folders = [os.path.join(source_folder, str(e)) for e in range(10)]
    result_folder = 'compare_measure'

    if os.path.isdir(result_folder):
        shutil.rmtree(result_folder)
    os.mkdir(result_folder)
        
    process_measure(config_folders, "accuracy", "Xent", result_folder)
    process_measure(config_folders, "true_accuracy", "Xent", result_folder)
    process_measure(config_folders, "f_measure", "Xent", result_folder)
    process_measure(config_folders, "true_accuracy", "accuracy", result_folder)
    process_measure(config_folders, "f_measure", "accuracy", result_folder)
    process_measure(config_folders, "f_measure", "true_accuracy", result_folder)