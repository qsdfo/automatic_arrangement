#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import keras
import numpy as np

from LOP.Utils.build_input import build_sequence

import time

DEBUG = False

def train(model,
		  piano_train, orchestra_train, train_index,
		  piano_valid, orchestra_valid, valid_index,
		  parameters, config_folder, start_time_train, logger_train):
   
	# Time information used
	time_limit = parameters['walltime'] * 3600 - 30*60  # walltime - 30 minutes in seconds

	############################################################
	# Compute train step
	preds = model.predict()
	# Declare labels placeholders
	labels = tf.placeholder(tf.float32, shape=(None, model.orchestra_dim), name="labels")
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
		if model.keras == True:
			from keras import backend as K
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
			train_monitor_epoch = []
			
			for batch_index in train_index:
				# Build batch
				piano_t = piano_train[batch_index]
				orchestra_past = build_sequence(orchestra_train, batch_index-1, model.batch_size, model.temporal_order-1, model.orchestra_dim)
				orchestra_t = orchestra_train[batch_index]
				
				# Train step
				logger_train.info("Cest parti")
				res = sess.run(train_step, {model.piano_t: piano_t,
											model.orchestra_past: orchestra_past,
											labels: orchestra_t,
											K.learning_phase(): 1})

		#      this_cost, this_monitor = model.train_batch(batch_data)
		#         # Keep track of cost
		#         train_cost_epoch.append(this_cost)
		#         train_monitor_epoch.append(this_monitor)
		#         # # Plot a random mean_chain
		#         # random_choice_mean_activation[:, :, batch_index] = mean_chain[:, ind_activation[batch_index], :]
		#         # # mean along batch axis
		#         # mean_activation[:, :, batch_index] = mean_chain.mean(axis=1)

		#     mean_loss = np.mean(train_cost_epoch)
		#     loss_tab[epoch] = mean_loss

		#     #######################################
		#     # Validate
		#     # For binary unit, it's an accuracy measure.
		#     # For real valued units its a gaussian centered value with variance 1
		#     #######################################
		#     accuracy = []
		#     for batch_index in valid_index:
		#         batch_data = model.generator(piano_valid, orchestra_valid, batch_index)
		#         # _, _, accuracy_batch, true_frame, past_frame, piano_frame, predicted_frame = validation_error(valid_index[batch_index])
		#         _, _, accuracy_batch = model.validate_batch(batch_data)
		#         accuracy += [accuracy_batch]

		#         # if parameters['DEBUG']:
		#             # from acidano.visualization.numpy_array.visualize_numpy import visualize_mat
		#             # if batch_index == 0:
		#             #     for ind in range(accuracy_batch.shape[0]):
		#             #         pr_viz = np.zeros((4, predicted_frame.shape[1]))
		#             #         # Threshold prediction
		#             #         orch_pred_ind = predicted_frame[ind]
		#             #         # Less than 1%
		#             #         thresh_pred = np.where(orch_pred_ind > 0.01, orch_pred_ind, 0)
		#             #         pr_viz[0] = thresh_pred
		#             #         pr_viz[1] = true_frame[ind]
		#             #         pr_viz[2] = past_frame[ind]
		#             #         pr_viz[3][:piano_frame.shape[1]] = piano_frame[ind]
		#             #         path_accuracy = config_folder + '/DEBUG/' + str(epoch) + '/validation'
		#             #         if not os.path.isdir(path_accuracy):
		#             #             os.makedirs(path_accuracy)
		#             #         visualize_mat(np.transpose(pr_viz), path_accuracy, str(ind) + '_score_' + str(accuracy_batch[ind]))
		#     mean_accuracy = 100 * np.mean(accuracy)

		#     end_time_epoch = time.time()

		#     #######################################
		#     # Is it the best model we have seen so far ?
		#     if mean_accuracy >= np.max(val_tab):
		#         best_model = model
		#         best_epoch = epoch
		#     #######################################
			
		#     #######################################
		#     # Article
		#     # Early stopping, but when ?
		#     # Lutz Prechelt
		#     # UP criterion (except that we expect accuracy to go up in our case,
		#     # so the minus sign)
		#     val_tab[epoch] = mean_accuracy
		#     if epoch >= parameters['min_number_iteration']:
		#         OVERFITTING = up_criterion(-val_tab, epoch, parameters["number_strips"], parameters["validation_order"])
		#     #######################################

		#     #######################################
		#     # Monitor time (guillimin walltime)
		#     if (time.time() - start_time_train) > time_limit:
		#         TIME_LIMIT = True
		#     #######################################

		#     #######################################
		#     # Log training
		#     #######################################
		#     logger_train.info(('Epoch : {} , Monitor : {} , Cost : {} , Valid acc : {}'
		#                       .format(epoch, np.mean(train_monitor_epoch), mean_loss, mean_accuracy))
		#                       .encode('utf8'))

		#     logger_train.info(('Time : {}'
		#                       .format(end_time_epoch - start_time_epoch))
		#                       .encode('utf8'))

		#     if OVERFITTING:
		#         logger_train.info('OVERFITTING !!')

		#     if TIME_LIMIT:
		#         logger_train.info('TIME OUT !!')

		#     #######################################
		#     # Epoch +1
		#     #######################################
		#     epoch += 1

		# # Return best accuracy
		# best_accuracy = val_tab[best_epoch]
		# best_loss = loss_tab[best_epoch]
		# return best_loss, best_accuracy, best_epoch, best_model
	return