#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras import backend as K
import numpy as np
import keras
import time
import os
from multiprocessing.pool import ThreadPool
# from multiprocessing.pool import Pool

import config
from LOP.Utils.early_stopping import up_criterion
from LOP.Utils.training_error import accuracy_low_TN_tf, bin_Xent_tf, bin_Xen_weighted_0_tf, accuracy_tf, sparsity_penalty_l1, sparsity_penalty_l2, bin_Xen_weighted_1_tf
from LOP.Utils.measure import accuracy_measure, precision_measure, recall_measure, true_accuracy_measure, f_measure, binary_cross_entropy
from LOP.Utils.build_batch import build_batch
from LOP.Utils.model_statistics import count_parameters
from LOP.Utils.Analysis.accuracy_and_binary_Xent import accuracy_and_binary_Xent
from LOP.Utils.Analysis.compare_Xent_acc_corresponding_preds import compare_Xent_acc_corresponding_preds

from asynchronous_load_mat import async_load_mat

DEBUG=True
# Note : debug sans summarize, qui pollue le tableau de variables
SUMMARIZE=False
ANALYSIS=False

def validate(context, init_matrices_validation, valid_splits_batches, valid_long_range_splits_batches, normalizer, parameters):
	
	sess = context['sess']
	temporal_order = context['temporal_order']
	mask_orch_ph = context['mask_orch_ph']
	inputs_ph = context['inputs_ph']
	orch_t_ph = context['orch_t_ph']
	preds_node = context['preds_node']
	loss_val_node = context['loss_val_node']
	keras_learning_phase = context['keras_learning_phase']
	
	#######################################
	# Validate
	#######################################
	piano_t_ph, piano_past_ph, piano_future_ph, orch_past_ph, orch_future_ph = inputs_ph
	accuracy = []
	precision = []
	recall = []
	val_loss = []
	true_accuracy = []
	f_score = []
	Xent = []
	if DEBUG:
		preds = []
		truth = []
	else:
		preds = None
		truth = None

	accuracy_long_range = []
	precision_long_range = []
	recall_long_range = []
	val_loss_long_range = []
	true_accuracy_long_range = []
	f_score_long_range = []
	Xent_long_range = []

	path_piano_matrices_valid = valid_splits_batches.keys()
	path_piano_matrices_valid_long_range = valid_long_range_splits_batches.keys()
	N_matrix_files = len(path_piano_matrices_valid)
	pool = ThreadPool(processes=1)
	matrices_from_thread = init_matrices_validation

	for file_ind_CURRENT in range(N_matrix_files):
		#######################################
		# Get indices and matrices to load
		#######################################
		# We train on the current matrix
		path_piano_matrix_CURRENT = path_piano_matrices_valid[file_ind_CURRENT]
		# Just a small check
		assert path_piano_matrix_CURRENT == path_piano_matrices_valid_long_range[file_ind_CURRENT], "Valid and valid_long_range are not the same"
		# Get indices
		valid_index = valid_splits_batches[path_piano_matrix_CURRENT]
		valid_long_range_index = valid_long_range_splits_batches[path_piano_matrix_CURRENT]
		# But load the next one : Useless if only one matrix, but I don't care, plenty of CPUs on lagavulin... (memory though ?)
		file_ind_NEXT = (file_ind_CURRENT+1) % N_matrix_files
		path_piano_matrix_NEXT = path_piano_matrices_valid[file_ind_NEXT]
		
		#######################################
		# Load matrix thread
		#######################################
		async_valid = pool.apply_async(async_load_mat, (normalizer, path_piano_matrix_NEXT, parameters))

		piano_transformed, orch, duration_piano, mask_orch = matrices_from_thread
	
		#######################################
		# Loop for short-term validation
		#######################################
		for batch_index in valid_index:
			# Build batch
			piano_t, piano_past, piano_future, orch_past, orch_future, orch_t, mask_orch_t = build_batch(batch_index, piano_transformed, orch, mask_orch, len(batch_index), temporal_order)
			# Input nodes
			feed_dict = {piano_t_ph: piano_t,
						piano_past_ph: piano_past,
						piano_future_ph: piano_future,
						orch_past_ph: orch_past,
						orch_future_ph: orch_future,
						orch_t_ph: orch_t,
						mask_orch_ph: mask_orch_t,
						keras_learning_phase: 0}

			# Compute validation loss
			preds_batch, loss_batch = sess.run([preds_node, loss_val_node], feed_dict)
			
			val_loss += [loss_batch] * len(batch_index) # Multiply by size of batch for mean : HACKY
			Xent_batch = binary_cross_entropy(orch_t, preds_batch)
			accuracy_batch = accuracy_measure(orch_t, preds_batch)
			precision_batch = precision_measure(orch_t, preds_batch)
			recall_batch = recall_measure(orch_t, preds_batch)
			true_accuracy_batch = true_accuracy_measure(orch_t, preds_batch)
			f_score_batch = f_measure(orch_t, preds_batch)
		
			accuracy.extend(accuracy_batch)
			precision.extend(precision_batch)
			recall.extend(recall_batch)
			true_accuracy.extend(true_accuracy_batch)
			f_score.extend(f_score_batch)
			Xent.extend(Xent_batch)

			if DEBUG:
				preds.extend(preds_batch)
				truth.extend(orch_t)

		#######################################
		# Loop for long-term validation
		# The task is filling a gap of size parameters["long_range"]
		# So the algo is given :
		#       orch[0:temporal_order]
		#       orch[temporal_order+parameters["long_range"]:
		#            (2*temporal_order)+parameters["long_range"]]
		# And must fill :
		#       orch[temporal_order:
		#            temporal_order+parameters["long_range"]]
		#######################################
		for batch_index in valid_long_range_index:
			# Init
			# Extract from piano and orchestra the matrices required for the task
			seq_len = (temporal_order-1) * 2 + parameters["long_range"]
			piano_dim = piano_transformed.shape[1]
			orch_dim = orch.shape[1]
			piano_extracted = np.zeros((len(batch_index), seq_len, piano_dim))
			orch_extracted = np.zeros((len(batch_index), seq_len, orch_dim))
			orch_gen = np.zeros((len(batch_index), seq_len, orch_dim))
			for ind_b, this_batch_ind in enumerate(batch_index):
				start_ind = this_batch_ind-temporal_order+1
				end_ind = start_ind + seq_len
				piano_extracted[ind_b] = piano_transformed[start_ind:end_ind,:]
				orch_extracted[ind_b] = orch[start_ind:end_ind,:]
			
			# We know the past orchestration at the beginning...
			orch_gen[:, :temporal_order-1, :] = orch_extracted[:, :temporal_order-1, :]
			# and the future orchestration at the end
			orch_gen[:, -temporal_order+1:, :] = orch_extracted[:, -temporal_order+1:, :]
			# check we didn't gave the correct information
			assert orch_gen[:, temporal_order-1:(temporal_order-1)+parameters["long_range"], :].sum()==0, "The gap to fill in orch_gen contains values !"

			for t in range(temporal_order-1, temporal_order-1+parameters["long_range"]):
				# We cannot use build_batch function here, but getting the matrices is quite easy
				piano_t = piano_extracted[:, t, :]
				piano_past = piano_extracted[:, t-(temporal_order-1):t, :]
				piano_future = piano_extracted[:, t+1:t+temporal_order, :]
				orch_t = orch_extracted[:, t, :]
				orch_past = orch_gen[:, t-(temporal_order-1):t, :]
				orch_future = orch_gen[:, t+1:t+temporal_order, :]
				mask_orch_t = np.ones_like(orch_t)

				# Feed dict
				feed_dict = {piano_t_ph: piano_t,
							piano_past_ph: piano_past,
							piano_future_ph: piano_future,
							orch_past_ph: orch_past,
							orch_future_ph: orch_future,
							orch_t_ph: orch_t,
							mask_orch_ph: mask_orch_t,
							keras_learning_phase: 0}

				# Get prediction
				preds_batch, loss_batch = sess.run([preds_node, loss_val_node], feed_dict)
				# Preds should be a probability distribution. Sample from it
				# Note that it doesn't need to be part of the graph since we don't use the sampled value to compute the backproped error
				
				prediction_sampled = np.random.binomial(1, preds_batch)
				orch_gen[:, t, :] = prediction_sampled

				# Compute performances measures
				val_loss_long_range += [loss_batch] * len(batch_index) # Multiply by size of batch for mean : HACKY
				Xent_batch = binary_cross_entropy(orch_t, preds_batch)
				accuracy_batch = accuracy_measure(orch_t, preds_batch)
				precision_batch = precision_measure(orch_t, preds_batch)
				recall_batch = recall_measure(orch_t, preds_batch)
				true_accuracy_batch = true_accuracy_measure(orch_t, preds_batch)
				f_score_batch = f_measure(orch_t, preds_batch)

				accuracy_long_range.extend(accuracy_batch)
				precision_long_range.extend(precision_batch)
				recall_long_range.extend(recall_batch)
				true_accuracy_long_range.extend(true_accuracy_batch)
				f_score_long_range.extend(f_score_batch)
				Xent_long_range.extend(Xent_batch)

		del(matrices_from_thread)
		matrices_from_thread = async_valid.get()

	pool.close()
	pool.join()
	valid_results = {
		'accuracy': np.asarray(accuracy), 
		'precision': np.asarray(precision), 
		'recall': np.asarray(recall), 
		'val_loss': np.asarray(val_loss), 
		'true_accuracy': np.asarray(true_accuracy), 
		'f_score': np.asarray(f_score), 
		'Xent': np.asarray(Xent)
		}
	valid_long_range_results = {
		'accuracy': np.asarray(accuracy_long_range),
		'precision': np.asarray(precision_long_range), 
		'recall': np.asarray(recall_long_range),
		'val_loss': np.asarray(val_loss_long_range), 
		'true_accuracy': np.asarray(true_accuracy_long_range), 
		'f_score': np.asarray(f_score_long_range), 
		'Xent': np.asarray(Xent_long_range)
		}
	return valid_results, valid_long_range_results, preds, truth
		

def train(model, train_splits_batches, valid_splits_batches, valid_long_range_splits_batches, normalizer,
		  parameters, config_folder, start_time_train, logger_train):

	# Time information used
	time_limit = parameters['walltime'] * 3600 - 30*60  # walltime - 30 minutes in seconds

	# Reset graph before starting training
	tf.reset_default_graph()
		
	###### PETIT TEST VALIDATION
	# Use same validation en train set
	# piano_valid, orch_valid, valid_index = piano_train, orch_train, train_index

	if parameters['pretrained_model'] is None:
		logger_train.info((u'#### Graph'))
		start_time_building_graph = time.time()
		inputs_ph, orch_t_ph, preds_node, loss_node, loss_val_node, mask_orch_ph, train_step_node, keras_learning_phase, debug, saver = build_training_nodes(model, parameters)
		time_building_graph = time.time() - start_time_building_graph
		logger_train.info("TTT : Building the graph took {0:.2f}s".format(time_building_graph))
	else:
		logger_train.info((u'#### Graph'))
		start_time_building_graph = time.time() 
		inputs_ph, orch_t_ph, preds_node, loss_node, loss_val_node, mask_orch_ph, train_step_node, keras_learning_phase, debug, saver = load_pretrained_model(parameters['pretrained_model'])
		time_building_graph = time.time() - start_time_building_graph
		logger_train.info("TTT : Loading pretrained model took {0:.2f}s".format(time_building_graph))

	piano_t_ph, piano_past_ph, piano_future_ph, orch_past_ph, orch_future_ph = inputs_ph
	embedding_concat = debug[0]
	sparse_loss_node = debug[1]

	if SUMMARIZE:
		tf.summary.scalar('loss', loss_node)
	############################################################

	############################################################
	# Display informations about the models
	num_parameters = count_parameters(tf.get_default_graph())
	logger_train.info((u'** Num trainable parameters :  {}'.format(num_parameters)).encode('utf8'))
	with open(os.path.join(config_folder, 'num_parameters.txt'), 'wb') as ff:
		ff.write("{:d}".format(num_parameters))

	############################################################
	# Training
	logger_train.info("#" * 60)
	logger_train.info("#### Training")
	epoch = 0
	OVERFITTING = False
	TIME_LIMIT = False

	# Train error
	loss_tab = np.zeros(max(1, parameters['max_iter']))

	# Select criteria
	overfitting_measure = parameters["overfitting_measure"]
	save_measures = parameters['save_measures']

	# Short-term validation error
	valid_tabs = {
		'loss': np.zeros(max(1, parameters['max_iter'])),
		'accuracy': np.zeros(max(1, parameters['max_iter'])),
		'precision': np.zeros(max(1, parameters['max_iter'])),
		'recall': np.zeros(max(1, parameters['max_iter'])),
		'true_accuracy': np.zeros(max(1, parameters['max_iter'])),
		'f_score': np.zeros(max(1, parameters['max_iter'])),
		'Xent': np.zeros(max(1, parameters['max_iter']))
		}
	# Best epoch for each measure
	best_epoch = {
		'loss': 0, 
		'accuracy': 0, 
		'precision': 0, 
		'recall': 0, 
		'true_accuracy': 0, 
		'f_score': 0, 
		'Xent': 0
	}
	
	# Long-term validation error
	valid_tabs_LR = {
		'loss': np.zeros(max(1, parameters['max_iter'])), 
		'accuracy': np.zeros(max(1, parameters['max_iter'])), 
		'precision': np.zeros(max(1, parameters['max_iter'])), 
		'recall': np.zeros(max(1, parameters['max_iter'])), 
		'true_accuracy': np.zeros(max(1, parameters['max_iter'])), 
		'f_score': np.zeros(max(1, parameters['max_iter'])), 
		'Xent': np.zeros(max(1, parameters['max_iter']))
		}
	# Best epoch for each measure
	best_epoch_LR = {
		'loss': 0, 
		'accuracy': 0, 
		'precision': 0, 
		'recall': 0, 
		'true_accuracy': 0, 
		'f_score': 0, 
		'Xent': 0
	}

	if parameters['memory_gpu']:
		##############
		##############
		##############
		# This apparently does not work
		configSession = tf.ConfigProto()
		configSession.gpu_options.per_process_gpu_memory_fraction = parameters['memory_gpu']
		##############
		##############
		##############
	else:
		configSession = None

	with tf.Session(config=configSession) as sess:
		
		if SUMMARIZE:
			merged_node = tf.summary.merge_all()
			train_writer = tf.summary.FileWriter(config_folder + '/summary', sess.graph)

		if model.is_keras():
			K.set_session(sess)
		
		# Initialize weights
		if parameters['pretrained_model']: 
			saver.restore(sess, parameters['pretrained_model'])
		else:
			sess.run(tf.global_variables_initializer())
			

		# if DEBUG:
		# 	sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		# 	sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
			
		############################################################
		# Define the context dict used to store graphs and nodes and use them in auxiliary functions
		context = {}
		context['sess'] = sess
		context['temporal_order'] = model.temporal_order
		context['inputs_ph'] = inputs_ph
		context['orch_t_ph'] = orch_t_ph
		context['mask_orch_ph'] = mask_orch_ph
		context['preds_node'] = preds_node
		context['loss_node'] = loss_node
		context['loss_val_node'] = loss_val_node
		context['keras_learning_phase'] = keras_learning_phase
		############################################################
		
		#######################################
		# Load first matrix
		#######################################
		path_piano_matrices_train = train_splits_batches.keys()
		N_matrix_files = len(path_piano_matrices_train)

		global_time_start = time.time()
		
		load_data_start = time.time()
		pool = ThreadPool(processes=1)
		async_train = pool.apply_async(async_load_mat, (normalizer, path_piano_matrices_train[0], parameters))
		matrices_from_thread = async_train.get()
		init_matrices_validation = matrices_from_thread
		load_data_time = time.time() - load_data_start
		logger_train.info("Load the first matrix time : " + str(load_data_time))

		if model.optimize() == False:
			# Some baseline models don't need training step optimization
			valid_results, valid_long_range_results, _, _ = validate(context, init_matrices_validation, valid_splits_batches, valid_long_range_splits_batches, normalizer, parameters)
			mean_and_store_results(valid_results, valid_tabs, 0)
			mean_and_store_results(valid_long_range_results, valid_tabs_LR, 0)
			return remove_tail_training_curves(valid_tabs, 1), best_epoch, \
				remove_tail_training_curves(valid_tabs_LR, 1), best_epoch_LR

		# Training iteration
		while (not OVERFITTING and not TIME_LIMIT
			   and epoch != parameters['max_iter']):
		
			start_time_epoch = time.time()

			train_cost_epoch = []
			sparse_loss_epoch = []

			for file_ind_CURRENT in range(N_matrix_files):

				#######################################
				# Get indices and matrices to load
				#######################################
				# We train on the current matrix
				path_piano_matrix_CURRENT = path_piano_matrices_train[file_ind_CURRENT]
				train_index = train_splits_batches[path_piano_matrix_CURRENT]
				# But load the one next one
				file_ind_NEXT = (file_ind_CURRENT+1) % N_matrix_files
				path_piano_matrix_NEXT = path_piano_matrices_train[file_ind_NEXT]
				
				#######################################
				# Load matrix thread
				#######################################
				async_train = pool.apply_async(async_load_mat, (normalizer, path_piano_matrix_NEXT, parameters))

				piano_transformed, orch, duration_piano, mask_orch = matrices_from_thread

				#######################################
				# Train
				#######################################
				for batch_index in train_index:
					# Build batch
					piano_t, piano_past, piano_future, orch_past, orch_future, orch_t, mask_orch_t = build_batch(batch_index, piano_transformed, orch, mask_orch, len(batch_index), model.temporal_order)

					# Train step
					feed_dict = {piano_t_ph: piano_t,
								piano_past_ph: piano_past,
								piano_future_ph: piano_future,
								orch_past_ph: orch_past,
								orch_future_ph: orch_future,
								orch_t_ph: orch_t,
								mask_orch_ph: mask_orch_t,
								keras_learning_phase: 1}

					if SUMMARIZE:
						_, loss_batch, summary = sess.run([train_step_node, loss_node, merged_node], feed_dict)
					else:
						_, loss_batch, preds_batch, sparse_loss_batch = sess.run([train_step_node, loss_node, preds_node, sparse_loss_node], feed_dict)
						# _, loss_batch, preds_batch = sess.run([train_step_node, loss_node, preds_node], feed_dict)
						# Keep track of cost
						train_cost_epoch.append(loss_batch)
						sparse_loss_epoch.append(sparse_loss_batch)

				#######################################
				# New matrices from thread
				#######################################
				del(matrices_from_thread)
				matrices_from_thread = async_train.get()

			if SUMMARIZE:
				if (epoch<5) or (epoch%10==0):
					# Note that summarize here only look at the variables after the last batch of the epoch
					# If you want to look at all the batches, include it in 
					train_writer.add_summary(summary, epoch)
	 
			mean_loss = np.mean(train_cost_epoch)
			loss_tab[epoch] = mean_loss

			#######################################
			# Validate
			#######################################
			valid_results, valid_long_range_results, preds_val, truth_val = validate(context, init_matrices_validation, valid_splits_batches, valid_long_range_splits_batches, normalizer, parameters)
			mean_and_store_results(valid_results, valid_tabs, epoch)
			mean_and_store_results(valid_long_range_results, valid_tabs_LR, epoch)
			end_time_epoch = time.time()

			#######################################
			# DEBUG
			# Save numpy arrays of measures values
			if DEBUG:
				for measure_name, measure_tab in valid_tabs.iteritems():
					np.save(os.path.join(config_folder, measure_name + '.npy'), measure_tab)
				np.save(os.path.join(config_folder, 'preds.npy'), np.asarray(preds_val))
				np.save(os.path.join(config_folder, 'truth.npy'), np.asarray(truth_val))
			#######################################
			
			#######################################
			# Overfitting ? 
			if epoch >= parameters['min_number_iteration']:
				# Choose short/long range and the measure
				OVERFITTING = up_criterion(valid_tabs[overfitting_measure], epoch, parameters["number_strips"], parameters["validation_order"])
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
			logger_train.info(('Epoch : {} , Training loss : {} , Validation loss : {} \n \
Validation accuracy : {:.3f} %, precision : {:.3f} %, recall : {:.3f} % \n \
True_accuracy : {:.3f} %, f_score : {:.3f} %, Xent_100 : {:.3f}\n \
Sparse_loss : {:.3f}'
							  .format(epoch, mean_loss,
								valid_tabs['loss'][epoch], valid_tabs['accuracy'][epoch], valid_tabs['precision'][epoch],
								valid_tabs['recall'][epoch], valid_tabs['true_accuracy'][epoch], valid_tabs['f_score'][epoch], 100*valid_tabs['Xent'][epoch],
								np.mean(sparse_loss_epoch))
							  .encode('utf8')))

			logger_train.info(('Time : {}'
							  .format(end_time_epoch - start_time_epoch))
							  .encode('utf8'))

			#######################################
			# Best model ?
			# Xent criterion
			start_time_saving = time.time()
			for measure_name, measure_curve in valid_tabs.iteritems():
				best_measure_so_far = measure_curve[best_epoch[measure_name]]
				measure_for_this_epoch = measure_curve[epoch]
				if measure_for_this_epoch <= best_measure_so_far:
					if measure_name in save_measures:
						saver.save(sess, config_folder + "/model_" + measure_name + "/model")
					best_epoch[measure_name] = epoch
	   
			end_time_saving = time.time()
			logger_train.info('Saving time : {:.3f}'.format(end_time_saving-start_time_saving))
			#######################################

			if OVERFITTING:
				logger_train.info('OVERFITTING !!')

			if TIME_LIMIT:
				logger_train.info('TIME OUT !!')

			#######################################
			# Epoch +1
			#######################################
			epoch += 1

		pool.close()
		pool.join()
	
	return remove_tail_training_curves(valid_tabs, epoch), best_epoch, \
		remove_tail_training_curves(valid_tabs_LR, epoch), best_epoch_LR

def build_training_nodes(model, parameters):
	############################################################
	# Build nodes
	# Inputs
	piano_t_ph = tf.placeholder(tf.float32, shape=(None, model.piano_transformed_dim), name="piano_t")
	piano_past_ph = tf.placeholder(tf.float32, shape=(None, model.temporal_order-1, model.piano_transformed_dim), name="piano_past")
	piano_future_ph = tf.placeholder(tf.float32, shape=(None, model.temporal_order-1, model.piano_transformed_dim), name="piano_future")
	#
	orch_t_ph = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="orch_t")
	orch_past_ph = tf.placeholder(tf.float32, shape=(None, model.temporal_order-1, model.orch_dim), name="orch_past")
	orch_future_ph = tf.placeholder(tf.float32, shape=(None, model.temporal_order-1, model.orch_dim), name="orch_past")
	inputs_ph = (piano_t_ph, piano_past_ph, piano_future_ph, orch_past_ph, orch_future_ph)
	# Orchestral mask
	mask_orch_ph = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="mask_orch")
	# Prediction
	preds, embedding_concat = model.predict(inputs_ph)
	# TODO : remplacer cette ligne par une fonction qui prends labels et preds et qui compute la loss
	# Comme ça on pourra faire des classifier chains
	############################################################
	
	############################################################
	# Loss
	with tf.name_scope('loss'):
		# distance = keras.losses.binary_crossentropy(orch_t_ph, preds)
		# distance = Xent_tf(orch_t_ph, preds)
		# distance = bin_Xen_weighted_0_tf(orch_t_ph, preds, parameters['activation_ratio'])
		distance = bin_Xen_weighted_1_tf(orch_t_ph, preds, model.tn_weight)
		# distance = accuracy_tf(orch_t_ph, preds)
		# distance = accuracy_low_TN_tf(orch_t_ph, preds, weight=model.tn_weight)

		######################################################
		# Add sparsity constraint on the output ? Is it still loss_val or just loss :/ ???
		sparsity_coeff = model.sparsity_coeff
		sparse_loss = sparsity_penalty_l1(preds)
		# sparse_loss = sparsity_penalty_l2(preds)
		
		# Try something like this ???
		# sparse_loss = case({tf.less(sparse_loss, 10): (lambda: tf.constant(0))}, default=(lambda: sparse_loss), exclusive=True)
		# sparse_loss = tf.keras.layers.LeakyReLU(tf.reduce_sum(preds, axis=1))
		
		sparse_loss = sparsity_coeff * sparse_loss
		# DEBUG purposes
		sparse_loss_mean = tf.reduce_mean(sparse_loss)
		######################################################

		if sparsity_coeff != 0:
			loss_val_ = distance + sparse_loss
		else:
			loss_val_ = distance

		if parameters['mask_orch']:
			loss_masked = tf.where(mask_orch_ph==1, loss_val_, tf.zeros_like(loss_val_))
			loss_val = tf.reduce_mean(loss_masked, name="loss_val")
		else:
			loss_val = tf.reduce_mean(loss_val_, name="loss_val")
	
	# Weight decay 
	if model.weight_decay_coeff != 0:
		# Keras weight decay does not work...
		loss = loss_val + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * model.weight_decay_coeff
	else:
		loss = loss_val
	############################################################
	
	############################################################
	if model.optimize():
		# Some models don't need training
		train_step = config.optimizer().minimize(loss)
	else:
		train_step = None
		
	keras_learning_phase = K.learning_phase()
	
	############################################################
	# Saver
	tf.add_to_collection('preds', preds)
	tf.add_to_collection('orch_t_ph', orch_t_ph)
	tf.add_to_collection('loss', loss)
	tf.add_to_collection('loss_val', loss_val)
	tf.add_to_collection('mask_orch_ph', mask_orch_ph)
	tf.add_to_collection('train_step', train_step)
	tf.add_to_collection('keras_learning_phase', keras_learning_phase)
	tf.add_to_collection('inputs_ph', piano_t_ph)
	tf.add_to_collection('inputs_ph', piano_past_ph)
	tf.add_to_collection('inputs_ph', piano_future_ph)
	tf.add_to_collection('inputs_ph', orch_past_ph)
	tf.add_to_collection('inputs_ph', orch_future_ph)
	# Debug collection
	tf.add_to_collection('debug', embedding_concat)
	tf.add_to_collection('debug', sparse_loss_mean)
	debug = (embedding_concat, sparse_loss_mean)
	if model.optimize():
		saver = tf.train.Saver()
	else:
		saver = None
	############################################################
	
	return inputs_ph, orch_t_ph, preds, loss, loss_val, mask_orch_ph, train_step, keras_learning_phase, debug, saver

def load_pretrained_model(path_to_model):
	# Restore model and preds graph
	saver = tf.train.import_meta_graph(path_to_model + '.meta')
	inputs_ph = tf.get_collection('inputs_ph')
	orch_t_ph = tf.get_collection("orch_t_ph")[0]
	preds = tf.get_collection("preds")[0]
	loss = tf.get_collection("loss")[0]
	loss_val = tf.get_collection("loss_val")[0]
	mask_orch_ph = tf.get_collection("mask_orch_ph")[0]
	train_step = tf.get_collection('train_step')[0]
	keras_learning_phase = tf.get_collection("keras_learning_phase")[0]
	debug = tf.get_collection("debug")
	return inputs_ph, orch_t_ph, preds, loss, loss_val, mask_orch_ph, train_step, keras_learning_phase, debug, saver

def mean_and_store_results(results, tabs, epoch):
	# Use minus signs to ensure measures are loss
	mean_val_loss = np.mean(results['val_loss'])
	mean_accuracy = -100 * np.mean(results['accuracy'])
	mean_precision = -100 * np.mean(results['precision'])
	mean_recall = -100 * np.mean(results['recall'])
	mean_true_accuracy = -100 * np.mean(results['true_accuracy'])
	mean_f_score = -100 * np.mean(results['f_score'])
	mean_Xent = np.mean(results['Xent'])

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

# bias=[v.eval() for v in tf.global_variables() if v.name == "top_layer_prediction/orch_pred/bias:0"][0]
# kernel=[v.eval() for v in tf.global_variables() if v.name == "top_layer_prediction/orch_pred/kernel:0"][0]