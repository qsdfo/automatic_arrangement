#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:37:36 2017

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


def compare_Xent_acc_corresponding_preds(context, batches, save_folder):
    """Save te matrices for the accuracy score and Xent score and the corresponding ground-truth and predictions 

    """
    
    # Clean plot directory
    if os.path.isdir(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)
        
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
    
    np.save(os.path.join(save_folder, 'accuracy.npy'), acc)
    np.save(os.path.join(save_folder, 'Xent.npy'), 100 * neg_ll)
    np.save(os.path.join(save_folder, 'preds.npy'), preds_mat)
    np.save(os.path.join(save_folder, 'truth.npy'), truth_mat)

    return