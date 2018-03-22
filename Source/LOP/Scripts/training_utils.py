#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np


def mean_and_store_results(results, tabs, epoch):
	# Use minus signs to ensure measures are loss
	mean_val_loss = np.mean(results['loss'])
	mean_accuracy = -100 * np.mean(results['accuracy'])
	mean_precision = -100 * np.mean(results['precision'])
	mean_recall = -100 * np.mean(results['recall'])
	mean_true_accuracy = -100 * np.mean(results['true_accuracy'])
	mean_f_score = -100 * np.mean(results['f_score'])
	mean_Xent = np.mean(results['Xent'])

	if epoch is None:
		tabs['loss'] = mean_val_loss
		tabs['accuracy'] = mean_accuracy
		tabs['precision'] = mean_precision
		tabs['recall'] = mean_recall
		tabs['true_accuracy'] = mean_true_accuracy
		tabs['f_score'] = mean_f_score
		tabs['Xent'] = mean_Xent
	else:
		tabs['loss'][epoch] = mean_val_loss
		tabs['accuracy'][epoch] = mean_accuracy
		tabs['precision'][epoch] = mean_precision
		tabs['recall'][epoch] = mean_recall
		tabs['true_accuracy'][epoch] = mean_true_accuracy
		tabs['f_score'][epoch] = mean_f_score
		tabs['Xent'][epoch] = mean_Xent
	return


# Remove useless part of measures curves
def remove_tail_training_curves(dico, epoch):
	ret = {}
	for k, v in dico.iteritems():
		ret[k] = v[:epoch]
	return ret