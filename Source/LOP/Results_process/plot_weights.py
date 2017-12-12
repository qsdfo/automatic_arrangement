#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Script for plotting D3.js weights of a trained model
Created on Mon Dec 11 16:41:03 2017

@author: leo
"""

import os
import re
import shutil
import cPickle as pkl
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from LOP_database.visualization.numpy_array.visualize_numpy import visualize_mat


def plot_weights(model_name, path_config, path_plots):
    # Paths
    path_model = os.path.join(path_config, model_name)
    is_keras = pkl.load(open(path_config + '/../is_keras.pkl', 'rb'))
                
    # Restore model and preds graph
    tf.reset_default_graph() # First clear graph to avoid memory overflow when running training and generation in the same process
    saver = tf.train.import_meta_graph(path_model + '/model.meta')
    
    with tf.Session() as sess:            
        if is_keras:
            K.set_session(sess)    
        saver.restore(sess, path_model + '/model')

        # Plot weights with D3.js
        weight_folder = os.path.join(path_config, 'weights')
        if os.path.isdir(weight_folder):
            shutil.rmtree(weight_folder)
        os.mkdir(weight_folder)
        for trainable_parameter in tf.trainable_variables():
            name = trainable_parameter.name
            name = re.sub(':', '_', name)
            split_name = re.split('/', name)
            new_path = "/".join(split_name[:-1])
            new_path = os.path.join(weight_folder, new_path)
            new_name = split_name[-1]
            trainable_parameter_value = trainable_parameter.eval()
            tp_shape = trainable_parameter_value.shape
            num_param = 1
            for dim in tp_shape:
                num_param *= dim
            if num_param < (500*500):
                visualize_mat(trainable_parameter_value, new_path, new_name)
            else:
                if not os.path.isdir(new_path):
                    os.makedirs(new_path)
                plt.clf()
                plt.imshow(trainable_parameter_value, cmap='hot')
                plt.colorbar()
                plt.savefig(os.path.join(new_path, new_name + '.pdf'))
            
if __name__ == '__main__':
    # model_name = "model_acc"
    # root = "/Users/leo/Recherche/GitHub_Aciditeam/automatic_arrangement/Experiments/Fixed_static_biases/precomputed_fixed_static_biases_quali/"
    # path_configs = [
    #         root + "LSTM_plugged_base/0/0",
    #         root + "LSTM_plugged_base/1/0",
    #         root + "LSTM_static_bias/0/0",
    #         root + "LSTM_static_bias/1/0",
    #     ]
    # for path_config in path_configs:
    #     path_plots = os.path.join(path_config, "weights")
    #     plot_weights(model_name, path_config, path_plots)