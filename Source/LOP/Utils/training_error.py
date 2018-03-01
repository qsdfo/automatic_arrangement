#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Tensorflow implementations of several losses
Created on Mon Dec  4 16:09:35 2017

@author: leo
"""

import tensorflow as tf


def accuracy_low_TN_tf(true_frame, pred_frame, weight):
    """Modified accuracy function that includes the true negative (TN) but with a coefficient keep its influence low.
    
    """
    axis = len(true_frame.shape)-1
    true_positive = tf.reduce_sum(pred_frame * true_frame, axis)
    true_negative_weighted = weight * tf.reduce_sum(tf.multiply((1-pred_frame), (1-true_frame)), axis)
    false_negative = tf.reduce_sum(tf.multiply((1 - pred_frame), true_frame), axis)
    false_positive = tf.reduce_sum(tf.multiply(pred_frame, (1 - true_frame)), axis)

    quotient = true_positive + false_negative + false_positive + true_negative_weighted

    accuracy_measure = tf.div((true_negative_weighted + true_positive), quotient)

    return -accuracy_measure


def accuracy_tf(true_frame, pred_frame):
    """Modified accuracy function that includes the true negative (TN) but with a coefficient keep its influence low.
    
    """
    axis = len(true_frame.shape)-1
    epsilon = 1e-20
    true_positive = tf.reduce_sum(pred_frame * true_frame, axis)
    false_negative = tf.reduce_sum(tf.multiply((1 - pred_frame), true_frame), axis)
    false_positive = tf.reduce_sum(tf.multiply(pred_frame, (1 - true_frame)), axis)

    quotient = true_positive + false_negative + false_positive + epsilon

    accuracy_measure = tf.div((true_positive), quotient)

    return -100 * accuracy_measure

def bin_Xent_tf(true_frame, pred_frame):
    """Binary cross-entropy. Should be exactly the same as keras.losses.binary_crossentropy
    
    """
    axis = len(true_frame.shape)-1
    epsilon = 1e-20
    cross_entr_dot = tf.multiply(true_frame, tf.log(pred_frame+epsilon)) + tf.multiply((1-true_frame), tf.log((1-pred_frame+epsilon)))
    # Mean over feature dimension
    cross_entr = - tf.reduce_mean(cross_entr_dot, axis=axis)
    return cross_entr


def bin_Xent_NO_MEAN_tf(true_frame, pred_frame):
    """Binary cross-entropy. Should be exactly the same as keras.losses.binary_crossentropy
    
    """
    epsilon = 1e-20
    cross_entr_dot = tf.multiply(true_frame, tf.log(pred_frame+epsilon)) + tf.multiply((1-true_frame), tf.log((1-pred_frame+epsilon)))
    return -cross_entr_dot


def bin_Xen_weighted_0_tf(true_frame, pred_frame, activation_ratio):
    """Binary cross-entropy. Should be exactly the same as keras.losses.binary_crossentropy
    
    """
    axis = len(true_frame.shape)-1
    epsilon = 1e-20
    cross_entr_dot = (1-activation_ratio) * (true_frame * tf.log(pred_frame+epsilon)) + activation_ratio * (1-true_frame) * tf.log((1-pred_frame+epsilon))
    # Mean over feature dimension
    cross_entr = - tf.reduce_mean(cross_entr_dot, axis=axis)
    return cross_entr


def bin_Xen_weighted_1_tf(true_frame, pred_frame, weight_neg):
    """Binary cross-entropy. Should be exactly the same as keras.losses.binary_crossentropy
    
    """
    axis = len(true_frame.shape)-1
    epsilon = 1e-20
    cross_entr_dot = (true_frame * tf.log(pred_frame+epsilon)) + weight_neg * (1-true_frame) * tf.log((1-pred_frame+epsilon))
    # Mean over feature dimension
    cross_entr = - tf.reduce_mean(cross_entr_dot, axis=axis)
    return cross_entr

def sparsity_penalty_l1(proba_activation):
    axis = len(proba_activation.shape)-1
    # Abs should be useless as proba_activation is in [0,1], but it's a safeguard
    return tf.reduce_sum(tf.abs(proba_activation), axis=axis)

def sparsity_penalty_l2(proba_activation):
    axis = len(proba_activation.shape)-1
    squared_proba_activation = proba_activation * proba_activation
    return tf.reduce_sum(squared_proba_activation, axis=axis)
