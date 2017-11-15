#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:39:45 2017

@author: leo
"""

import os
import shutil
import numpy as np
import tensorflow as tf
import keras
from LOP.Utils.measure import accuracy_measure
from LOP_database.visualization.numpy_array.visualize_numpy import visualize_mat_proba
from LOP.Utils.build_batch import build_batch


def accuracy_and_binary_Xent(context, batches, plot_folder, N_example):
    """Used to compare what is considered good or bad for the accuracy and the neg-ll. Both values are computed over the whole test set.
    After normalization (mean and std), we plot the most representative examples of all the possible combinations of bad, average and good for neg-ll and acc
    
    Parameters
    ----------
    sess
        tensor flow session
    temporal_order
        temporal_order of the model, i.e. how far in the past/future is the model allowed to look
    batches
        a list containing lists of indices. Each list of indices is a mini-batch
    piano
        matrix containing the piano score
    orch
        matrix containing the orch score
    inputs_ph
        lists of placeholders which are inputs of the graph
    orch_t_ph
        placeholder for the true orchestral frame
    preds
        placeholder for the prediction of the network
    keras_learning_phase
        placeholder for a flag used by Keras

    Returns
    -------
    None
    
    """

    # Clean plot directory
    if os.path.isdir(plot_folder):
        shutil.rmtree(plot_folder)
    os.mkdir(plot_folder)
        
    # Unpack context variables    
    sess = context['sess']
    temporal_order = context['temporal_order']
    piano = context['piano']
    orch = context['orch']
    inputs_ph = context['inputs_ph']
    orch_t_ph = context['orch_t_ph']
    preds = context['preds']
    keras_learning_phase = context['keras_learning_phase']
    
    piano_t_ph, piano_past_ph, piano_future_ph, orch_past_ph, orch_future_ph = inputs_ph
    all_preds = []
    all_truth = []
    # Get all the predictions for the whole set
    for batch_index in batches:
        # Build batch
        piano_t, piano_past, piano_future, orch_past, orch_future, orch_t = build_batch(batch_index, piano, orch, len(batch_index), temporal_order)
        # Input nodes
        feed_dict = {piano_t_ph: piano_t,
                    piano_past_ph: piano_past,
                    piano_future_ph: piano_future,
                    orch_past_ph: orch_past,
                    orch_future_ph: orch_future,
                    orch_t_ph: orch_t,
                    keras_learning_phase: 0}
        # Compute validation loss
        preds_batch = sess.run(preds, feed_dict)
        all_preds.extend(preds_batch)
        all_truth.extend(orch_t)
        
    # Cast as np array
    preds_mat = np.asarray(all_preds)
    truth_mat = np.asarray(all_truth)
    
    # Compute neg-ll (use keras function for coherency with validation step)
    truth_mat_ph = tf.placeholder(tf.float32, shape=(truth_mat.shape), name="truth_mat")
    preds_mat_ph = tf.placeholder(tf.float32, shape=(preds_mat.shape), name="preds_mat")
    neg_ll_node = keras.losses.binary_crossentropy(truth_mat_ph, preds_mat_ph)
    neg_ll = sess.run(neg_ll_node, {preds_mat_ph: preds_mat, truth_mat_ph: truth_mat})
    
    # Compute acc
    acc = 100 * accuracy_measure(truth_mat, preds_mat)
    
    # Normalize
    def normalize(matrix):
        mean = np.mean(matrix)
        std = np.std(matrix)
        norm_mat = (matrix - mean) / std
        return norm_mat, mean, std 
    neg_ll_norm, neg_ll_mean, neg_ll_std = normalize(neg_ll)
    acc_norm, acc_mean, acc_std = normalize(acc)
    
    # Rank perf
    arg_neg_ll = np.argsort(neg_ll_norm)
    arg_acc = np.argsort(-acc_norm) # minus to have the best at the first index
    num_index = len(arg_neg_ll)
    
    # Plots
    def plot(ind, this_folder):
        temp_mat = np.stack((truth_mat[ind], preds_mat[ind]))
        visualize_mat_proba(temp_mat, this_folder, "acc_" + str(acc[ind]) + "_nll_" + str(neg_ll[ind]))
    
    # Reste juste à parser les arg_sorted pour avoir les bons et mauvais exemples
    for index in range(N_example):            
        good_index = index
        bad_index = -index-1
        average_index = (num_index / 2) + (1 - 2 * (index % 2)) * index
        # Bad Xent
        bad_xent_folder = os.path.join(plot_folder, "bad_Xent_" + str(index))
        plot(arg_neg_ll[bad_index], bad_xent_folder)        
        # Average Xent
        plot(arg_neg_ll[average_index], os.path.join(plot_folder, "average_Xent_" + str(index)))
        # Good Xent
        plot(arg_neg_ll[good_index], os.path.join(plot_folder, "good_Xent_" + str(index)))
        # Bad acc
        plot(arg_acc[bad_index], os.path.join(plot_folder, "bad_acc_" + str(index)))        
        # Average acc
        plot(arg_acc[average_index], os.path.join(plot_folder, "average_acc_" + str(index)))
        # Good acc
        plot(arg_acc[good_index], os.path.join(plot_folder, "good_acc_" + str(index)))
    
    # Write the statistics
    with open(os.path.join(plot_folder, "statistics.txt"), "wb") as f:
        f.write("Accuracy mean : " + str(acc_mean) + "\n")
        f.write("Accuracy std : " + str(acc_std) + "\n")
        f.write("Xent mean : " + str(neg_ll_mean) + "\n")
        f.write("Xent std : " + str(neg_ll_std) + "\n")    
    return