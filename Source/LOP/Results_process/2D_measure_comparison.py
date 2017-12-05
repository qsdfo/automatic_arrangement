#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Compare the surfaces of the different measures in a 2 dimensional case.
Created on Wed Nov 22 15:30:57 2017

@author: leo
"""

import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from LOP.Utils.measure import accuracy_measure, accuracy_measure_test, true_accuracy_measure, f_measure, binary_cross_entropy, accuracy_measure_test_2

def plot_measure_2D(measure_fun, step_size, result_folder):

    fig = plt.figure()
    
    # Make data, avoiding 0 and 1 which raise problems
    p1 = np.arange(step_size, 1-step_size, step_size) 
    p2 = np.arange(step_size, 1-step_size, step_size)
    N_dim = len(p1)
    X, Y = np.meshgrid(p1, p2)
    X_reshape = np.reshape(X, [-1,])
    Y_reshape = np.reshape(Y, [-1,])
    pred_frame_mat = np.stack((X_reshape,Y_reshape), axis=1)
    
    for ind, true_frame in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        
        ax = fig.add_subplot(2, 2, ind+1, projection='3d')
    
        true_frame_ = np.reshape(true_frame, [-1, 1])
        true_frame_mat = (true_frame_.repeat(N_dim * N_dim, 1)).T
        measurement = measure_fun(true_frame_mat, pred_frame_mat)
        Z = np.reshape(measurement, [N_dim, N_dim])
        
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        
        # Customize the z axis.
        ax.set_zlim(Z.min()-0.1, Z.max()+0.1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
    plt.savefig(result_folder + '/' + measure_fun.__name__ + '__' + str(true_frame) + '.pdf')
    plt.show()


if __name__ == '__main__':
    result_dir = "plot_measure_2D"
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
        
#    plot_measure_2D(accuracy_measure, 0.01, result_dir)
#    plot_measure_2D(binary_cross_entropy, 0.01, result_dir)
#    plot_measure_2D(true_accuracy_measure, (0,1), 0.01, result_dir)
#    plot_measure_2D(f_measure, (0,1), 0.01, result_dir)
#    plot_measure_2D(binary_cross_entropy, (0,1), 0.01, result_dir)
    
    plot_measure_2D(accuracy_measure_test, 0.01, result_dir)
#    plot_measure_2D(accuracy_measure_test_2, 0.01, result_dir)