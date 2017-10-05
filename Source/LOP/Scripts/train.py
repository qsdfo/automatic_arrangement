#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import keras
from keras import backend as K
import numpy as np
import time
import os

from LOP.Utils.early_stopping import up_criterion
from LOP.Utils.measure import accuracy_measure, precision_measure, recall_measure
from LOP.Utils.build_batch import build_batch
from LOP.Utils.get_statistics import count_parameters

DEBUG = False
# Note : debug sans summarize, qui pollue le tableau de variables
SUMMARIZE = False
# Device to use
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Logging deive use ?
LOGGING_DEVICE = False

def train(model,
		  piano_train, orch_train, train_index,
		  piano_valid, orch_valid, valid_index,
		  parameters, config_folder, start_time_train, logger_train):
   
	# Time information used
	time_limit = parameters['walltime'] * 3600 - 30*60  # walltime - 30 minutes in seconds

	# Reset graph before starting training
	tf.reset_default_graph()
		
	###### PETIT TEST VALIDATION
	# Use same validation en train set
	# piano_valid, orch_valid, valid_index = piano_train, orch_train, train_index

	############################################################
	# Compute train step
	# Inputs
	logger_train.info((u'#### Graph'))
	start_time_building_graph = time.time()
	piano_t_ph = tf.placeholder(tf.float32, shape=(None, model.piano_dim), name="piano_t")
	orch_past_ph = tf.placeholder(tf.float32, shape=(None, model.temporal_order-1, model.orch_dim), name="orch_past")	
	# Prediction
	preds = model.predict(piano_t_ph, orch_past_ph)
	# Declare labels placeholders
	labels_ph = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="labels")
	# TODO : remplacer cette ligne par une fonction qui prends labels et preds et qui compute la loss
	# Comme ça on pourra faire des classifier chains
	loss = tf.reduce_mean(keras.losses.binary_crossentropy(labels_ph, preds), name="loss")
	# train_step = tf.train.AdamOptimizer(0.5).minimize(loss)
	train_step = tf.train.AdamOptimizer().minimize(loss)
	time_building_graph = time.time() - start_time_building_graph
	logger_train.info("TTT : Building the graph took {0:.2f}s".format(time_building_graph))
	############################################################

	############################################################
	# Saver
	# tf.add_to_collection('inputs', piano_t_ph)
	# tf.add_to_collection('inputs', orch_past_ph)
	tf.add_to_collection('preds', preds)
	saver = tf.train.Saver()
	############################################################

	############################################################
	# Display informations about the model
	num_parameters = count_parameters(tf.get_default_graph())
	logger_train.info((u'** Num trainable parameters :  {}'.format(num_parameters)).encode('utf8'))

	############################################################
	# Training
	logger_train.info("#" * 60)
	logger_train.info("#### Training")
	epoch = 0
	OVERFITTING = False
	TIME_LIMIT = False
	val_tab_acc = np.zeros(max(1, parameters['max_iter']))
	val_tab_prec = np.zeros(max(1, parameters['max_iter']))
	val_tab_rec = np.zeros(max(1, parameters['max_iter']))
	val_tab_loss = np.zeros(max(1, parameters['max_iter']))
	loss_tab = np.zeros(max(1, parameters['max_iter']))
	best_val_loss = float("inf")
	best_model = None
	best_epoch = None

	# with tf.Session(config=tf.ConfigProto(log_device_placement=LOGGING_DEVICE)) as sess:        
	with tf.Session() as sess:

		if SUMMARIZE: 
			merged = tf.summary.merge_all()
			train_writer = tf.summary.FileWriter(config_folder + '/summary', sess.graph)

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
				piano_t, orch_past, orch_t = build_batch(batch_index, piano_train, orch_train, model.batch_size, model.temporal_order)
				
				# Train step
				feed_dict = {piano_t_ph: piano_t,
							orch_past_ph: orch_past,
							labels_ph: orch_t,
							K.learning_phase(): 1}

				if SUMMARIZE:
					_, loss_batch, summary = sess.run([train_step, loss, merged], feed_dict)
				else:
					_, loss_batch = sess.run([train_step, loss], feed_dict)

				# Keep track of cost
				train_cost_epoch.append(loss_batch)

			if SUMMARIZE:
				if (epoch<5) or (epoch%10==0):
					train_writer.add_summary(summary, epoch)
 
			mean_loss = np.mean(train_cost_epoch)
			loss_tab[epoch] = mean_loss

			#######################################
			# Validate
			#######################################
			accuracy = []
			precision = []
			recall = []
			val_loss = []
			for batch_index in valid_index:
				# Build batch
				piano_t, orch_past, orch_t = build_batch(batch_index, piano_valid, orch_valid, model.batch_size, model.temporal_order)

				# Train step
				feed_dict = {piano_t_ph: piano_t,
							orch_past_ph: orch_past,
							labels_ph: orch_t,
							K.learning_phase(): 0}

				preds_batch, loss_batch = sess.run([preds, loss], feed_dict)
				val_loss += [loss_batch]
				accuracy_batch = accuracy_measure(orch_t, preds_batch)
				precision_batch = precision_measure(orch_t, preds_batch)
				recall_batch = recall_measure(orch_t, preds_batch)
				accuracy += [accuracy_batch]
				precision += [precision_batch]
				recall += [recall_batch]


			mean_val_loss = np.mean(val_loss)
			val_tab_loss[epoch] = mean_val_loss
			mean_accuracy = 100 * np.mean(accuracy)
			mean_precision = 100 * np.mean(precision)
			mean_recall = 100 * np.mean(recall)
			val_tab_acc[epoch] = mean_accuracy
			val_tab_prec[epoch] = mean_precision
			val_tab_rec[epoch] = mean_recall

			end_time_epoch = time.time()
			
			#######################################
			# Overfitting ?
			if epoch >= parameters['min_number_iteration']:
				OVERFITTING = up_criterion(val_tab_loss, epoch, parameters["number_strips"], parameters["validation_order"])
			#######################################

			#######################################
			# Monitor time (guillimin walltime)
			if (time.time() - start_time_train) > time_limit:
				TIME_LIMIT = True
			#######################################

			#######################################
			# Log training
			#######################################
			logger_train.info("############################################################")
			logger_train.info(('Epoch : {} , Training loss : {} , Validation binary_crossentr : {}, Validation accuracy : {} %, precision : {} %, recall : {} %'
							  .format(epoch, mean_loss, mean_val_loss, mean_accuracy, mean_precision, mean_recall))
							  .encode('utf8'))

			logger_train.info(('Time : {}'
							  .format(end_time_epoch - start_time_epoch))
							  .encode('utf8'))

			#######################################
			# Best model ?
			# if mean_accuracy >= np.max(val_tab_acc):
			if mean_val_loss <= best_val_loss:
				save_time_start = time.time()
				save_path = saver.save(sess, config_folder + "/model/model")
				best_epoch = epoch
				save_time = time.time() - save_time_start
				logger_train.info(('Save time : {}'.format(save_time)).encode('utf8'))
				best_val_loss = mean_val_loss
			#######################################

			if OVERFITTING:
				logger_train.info('OVERFITTING !!')

			if TIME_LIMIT:
				logger_train.info('TIME OUT !!')

			#######################################
			# Epoch +1
			#######################################
			epoch += 1

		# Return best accuracy
		best_accuracy = val_tab_acc[best_epoch]
		best_validation_loss = val_tab_loss[best_epoch]

	return best_validation_loss, best_accuracy, best_epoch
