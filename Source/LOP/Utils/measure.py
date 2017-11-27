#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np


def accuracy_measure(true_frame, pred_frame):
    axis = len(true_frame.shape) - 1

    true_positive = np.sum(pred_frame * true_frame, axis=axis)
    false_negative = np.sum((1 - pred_frame) * true_frame, axis=axis)
    false_positive = np.sum(pred_frame * (1 - true_frame), axis=axis)

    quotient = true_positive + false_negative + false_positive

    accuracy_measure = np.where(np.equal(quotient, 0), 0, np.true_divide(true_positive, quotient))

    return accuracy_measure


def accuracy_measure_test(true_frame, pred_frame):
    axis = len(true_frame.shape) - 1

    C = 0
    K = 0
    
    true_positive = np.sum(pred_frame * true_frame, axis=axis)
    false_negative = np.sum((1 - pred_frame) * true_frame, axis=axis)
    false_positive = np.sum(pred_frame * (1 - true_frame), axis=axis)

    quotient = true_positive + false_negative + false_positive
    
    accuracy_measure = np.where(np.equal(quotient, 0), 0, np.true_divide(true_positive + C, quotient + K))

    return accuracy_measure


def true_accuracy_measure(true_frame, pred_frame):
    axis = len(true_frame.shape) - 1

    true_positive = np.sum(pred_frame * true_frame, axis=axis)
    true_negative = np.sum((1-pred_frame) * (1-true_frame), axis=axis)
    false_negative = np.sum((1 - pred_frame) * true_frame, axis=axis)
    false_positive = np.sum(pred_frame * (1 - true_frame), axis=axis)

    quotient = true_positive + false_negative + false_positive + true_negative

    accuracy_measure = np.true_divide((true_negative + true_positive), quotient)

    return accuracy_measure


def f_measure(true_frame, pred_frame):
    precision = precision_measure(true_frame, pred_frame)
    recall = recall_measure(true_frame, pred_frame)
    f_measure = 2 * np.true_divide((precision * recall), (precision + recall))
    return f_measure


def accuracy_measure_continuous(true_frame, pred_frame):
    threshold = 0.05
    diff_abs = np.absolute(true_frame - pred_frame)
    diff_thresh = np.where(diff_abs < threshold, 1, 0)
    binary_truth = np.where(pred_frame > 0, 1, 0)
    tp = np.sum(diff_thresh * binary_truth, axis=-1)
    import pdb; pdb.set_trace()
    false = np.sum((1 - diff_thresh), axis=-1)

    quotient = tp + false

    accuracy_measure =  np.where(np.equal(quotient, 0), 0, np.true_divide(tp, quotient))

    return accuracy_measure


def recall_measure(true_frame, pred_frame):
    axis = len(true_frame.shape) - 1
    # true_frame must be a binary vector
    true_positive = np.sum(pred_frame * true_frame, axis=axis)
    false_negative = np.sum((1 - pred_frame) * true_frame, axis=axis)

    quotient = true_positive + false_negative

    recall_measure = np.where(quotient == 0, 0, true_positive / quotient)

    return recall_measure


def precision_measure(true_frame, pred_frame):
    axis = true_frame.ndim - 1
    # true_frame must be a binary vector
    true_positive = np.sum(pred_frame * true_frame, axis=axis)
    false_positive = np.sum(pred_frame * (1 - true_frame), axis=axis)

    quotient = true_positive + false_positive

    precision_measure = np.where(quotient == 0, 0, true_positive / quotient)

    return precision_measure


def binary_cross_entropy(true_frame, pred_frame):
    axis = true_frame.ndim - 1
    epsilon = 0.00001
    cross_entr_dot = np.multiply(true_frame, np.log(pred_frame+epsilon)) + np.multiply((1-true_frame), np.log((1-pred_frame+epsilon)))
    # Mean over feature dimension
    cross_entr = - np.mean(cross_entr_dot, axis=axis)
    return cross_entr