#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import keras
from keras import backend as K
import numpy as np

from LOP.Utils.build_input import build_sequence
from LOP.Utils.early_stopping import up_criterion

import time

DEBUG = False

def train(model,
		  piano_train, orch_train, train_index,
		  piano_valid, orch_valid, valid_index,
		  parameters, config_folder, start_time_train, logger_train):
   
	# Time information used
	time_limit = parameters['walltime'] * 3600 - 30*60  # walltime - 30 minutes in seconds

	############################################################
	# Compute train step
	preds = model.predict()
	# Declare labels placeholders
	labels = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="labels")
	# TODO : remplacer cette ligne par une fonction qui prends labels et preds et qui compute la loss
	# Comme ça on pourra faire des classifier chains
	loss = tf.reduce_mean(keras.losses.binary_crossentropy(labels, preds), name="loss")
	# train_step = tf.train.AdamOptimizer(0.5).minimize(loss)
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	############################################################

	############################################################
	# Training
	logger_train.info("#")
	logger_train.info("# Training")
	epoch = 0
	OVERFITTING = False
	TIME_LIMIT = False
	val_tab = np.zeros(max(1, parameters['max_iter']))
	loss_tab = np.zeros(max(1, parameters['max_iter']))
	best_model = None
	best_epoch = None

	with tf.Session() as sess:        

		writer = tf.summary.FileWriter("graph", sess.graph)

		if model.keras == True:
			K.set_session(sess)

		# Initialize weights
		sess.run(tf.global_variables_initializer())

		if DEBUG:
			sess = tf_debug.LocalCLIDebugWrapperSession(sess)
			sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

		# Training iteration
		while (not OVERFITTING and not TIME_LIMIT
			   and epoch != parameters['max_iter']):

			start_time_epoch = time.time()

			#######################################
			# Train
			#######################################
			train_cost_epoch = []			
			for batch_index in train_index:
				# Build batch
				piano_t, orch_past, orch_t = build_batch(batch_index, piano_train, orch_train, model.batch_size, model.temporal_order, model.orch_dim)
				
				# Train step
				feed_dict = {model.piano_t: piano_t,
							model.orch_past: orch_past,
							labels: orch_t,
							K.learning_phase(): 1}

				_, loss_batch = sess.run([train_step, loss], feed_dict)

				# Keep track of cost
				train_cost_epoch.append(loss_batch)
 
			mean_loss = np.mean(train_cost_epoch)
			loss_tab[epoch] = mean_loss

			#######################################
			# Validate
			#######################################
			accuracy = []
			for batch_index in valid_index:
				# Build batch
				piano_t, orch_past, orch_t = build_batch(batch_index, piano_valid, orch_valid, model.batch_size, model.temporal_order, model.orch_dim)

				# Train step
				feed_dict = {model.piano_t: piano_t,
							model.orch_past: orch_past,
							labels: orch_t,
							K.learning_phase(): 0}

				loss_batch = sess.run(loss, feed_dict)
				accuracy_batch = -loss_batch
				accuracy += [accuracy_batch]

			mean_accuracy = 100 * np.mean(accuracy)

			end_time_epoch = time.time()

			#######################################
			# Best model ?
			if mean_accuracy >= np.max(val_tab):
				best_model = model
				best_epoch = epoch
			#######################################
			
			#######################################
			# Overfitting ?
			val_tab[epoch] = mean_accuracy
			if epoch >= parameters['min_number_iteration']:
				OVERFITTING = up_criterion(-val_tab, epoch, parameters["number_strips"], parameters["validation_order"])
			#######################################

			#######################################
			# Monitor time (guillimin walltime)
			if (time.time() - start_time_train) > time_limit:
				TIME_LIMIT = True
			#######################################

			#######################################
			# Log training
			#######################################
			logger_train.info(('Epoch : {} , Training loss : {} , Validation score : {}'
							  .format(epoch, mean_loss, mean_accuracy))
							  .encode('utf8'))

			logger_train.info(('Time : {}'
							  .format(end_time_epoch - start_time_epoch))
							  .encode('utf8'))

			if OVERFITTING:
				logger_train.info('OVERFITTING !!')

			if TIME_LIMIT:
				logger_train.info('TIME OUT !!')

			#######################################
			# Epoch +1
			#######################################
			epoch += 1

		# # Return best accuracy
		# best_accuracy = val_tab[best_epoch]
		# best_loss = loss_tab[best_epoch]
		# return best_loss, best_accuracy, best_epoch, best_model
	return


def build_batch(batch_index, piano, orch, batch_size, temporal_order, orch_dim):
	# Build batch
	piano_t = piano[batch_index]
	orch_past = build_sequence(orch, batch_index-1, batch_size, temporal_order-1, orch_dim)
	orch_t = orch[batch_index]
	return piano_t, orch_past, orch_t