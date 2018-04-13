#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np
import keras
from keras import backend as K

import LOP.Scripts.config as config
from LOP.Utils.training_error import accuracy_low_TN_tf, bin_Xent_tf, bin_Xen_weighted_0_tf, accuracy_tf, sparsity_penalty_l1, sparsity_penalty_l2, bin_Xen_weighted_1_tf
from LOP.Utils.build_batch import build_batch

class Standard_trainer(object):
	
	def __init__(self, **kwargs):
		self.temporal_order = kwargs["temporal_order"]
		return

	def build_variables_nodes(self, model, parameters):
		# Build nodes
		# Inputs
		self.piano_t_ph = tf.placeholder(tf.float32, shape=(None, model.piano_transformed_dim), name="piano_t")
		self.piano_past_ph = tf.placeholder(tf.float32, shape=(None, self.temporal_order-1, model.piano_transformed_dim), name="piano_past")
		self.piano_future_ph = tf.placeholder(tf.float32, shape=(None, self.temporal_order-1, model.piano_transformed_dim), name="piano_future")
		#
		self.orch_t_ph = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="orch_t")
		self.orch_past_ph = tf.placeholder(tf.float32, shape=(None, self.temporal_order-1, model.orch_dim), name="orch_past")
		self.orch_future_ph = tf.placeholder(tf.float32, shape=(None, self.temporal_order-1, model.orch_dim), name="orch_past")
		# Orchestral mask
		self.mask_orch_ph = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="mask_orch")
		return

	def build_preds_nodes(self, model):
		inputs_ph = (self.piano_t_ph, self.piano_past_ph, self.piano_future_ph, self.orch_past_ph, self.orch_future_ph)
		# Prediction
		self.preds, self.embedding_concat = model.predict(inputs_ph)
		return
	
	def build_distance(self, model, parameters):
		distance = keras.losses.binary_crossentropy(self.orch_t_ph, self.preds)
		# distance = tf.losses.hinge_loss(self.orch_t_ph, self.preds)
		# distance = Xent_tf(orch_t_ph, self.preds)
		# distance = bin_Xen_weighted_0_tf(orch_t_ph, self.preds, parameters['activation_ratio'])
		# distance = bin_Xen_weighted_1_tf(self.orch_t_ph, self.preds, model.tn_weight)
		# distance = accuracy_tf(self.orch_t_ph, self.preds)
		# distance = accuracy_low_TN_tf(orch_t_ph, self.preds, weight=model.tn_weight)
		return distance
	
	def build_sparsity_term(self, model, parameters):
		# Add sparsity constraint on the output ? Is it still loss_val or just loss :/ ???
		sparsity_coeff = model.sparsity_coeff
		sparse_loss = sparsity_penalty_l1(self.preds)
		# sparse_loss = sparsity_penalty_l2(self.preds)
		
		# Try something like this ???
		# sparse_loss = case({tf.less(sparse_loss, 10): (lambda: tf.constant(0))}, default=(lambda: sparse_loss), exclusive=True)
		# sparse_loss = tf.keras.layers.LeakyReLU(tf.reduce_sum(self.preds, axis=1))
		
		sparse_loss = sparsity_coeff * sparse_loss
		# DEBUG purposes
		sparse_loss_mean = tf.reduce_mean(sparse_loss)
		return sparse_loss, sparse_loss_mean

	def build_weight_decay(self, model):
		weight_decay = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * model.weight_decay_coeff
		return weight_decay

	def build_loss_nodes(self, model, parameters):
		with tf.name_scope('loss'):
			with tf.name_scope('distance'):
				distance = self.build_distance(model, parameters)
		
			if model.sparsity_coeff != 0:
				with tf.name_scope('sparse_output_constraint'):
					sparse_loss, sparse_loss_mean = self.build_sparsity_term(model, parameters)
					loss_val_ = distance + sparse_loss
					self.sparse_loss_mean = sparse_loss_mean
			else:
				loss_val_ = distance
				temp = tf.zeros_like(loss_val_)
				self.sparse_loss_mean = tf.reduce_mean(temp)

			# Note: don't reduce_mean the validation loss, we need to have the per-sample value
			if parameters['mask_orch']:
				with tf.name_scope('mask_orch'):
					loss_masked = tf.where(mask_orch_ph==1, loss_val_, tf.zeros_like(loss_val_))
					self.loss_val = loss_masked
			else:
				self.loss_val = loss_val_
		
			mean_loss = tf.reduce_mean(self.loss_val)
			if model.weight_decay_coeff != 0:
				with tf.name_scope('weight_decay'):
					weight_decay = Standard_trainer.build_weight_decay(self, model)
					# Keras weight decay does not work...
					self.loss = mean_loss + weight_decay
			else:
				self.loss = mean_loss
		return
		
	def build_train_step_node(self, model, optimizer):
		if model.optimize():
			# Some models don't need training
			self.train_step = optimizer.minimize(self.loss)
		else:
			self.train_step = None
		self.keras_learning_phase = K.learning_phase()
		return
	
	def save_nodes(self, model):
		tf.add_to_collection('preds', self.preds)
		tf.add_to_collection('orch_t_ph', self.orch_t_ph)
		tf.add_to_collection('loss', self.loss)
		tf.add_to_collection('loss_val', self.loss_val)
		tf.add_to_collection('mask_orch_ph', self.mask_orch_ph)
		tf.add_to_collection('train_step', self.train_step)
		tf.add_to_collection('keras_learning_phase', self.keras_learning_phase)
		tf.add_to_collection('inputs_ph', self.piano_t_ph)
		tf.add_to_collection('inputs_ph', self.piano_past_ph)
		tf.add_to_collection('inputs_ph', self.piano_future_ph)
		tf.add_to_collection('inputs_ph', self.orch_past_ph)
		tf.add_to_collection('inputs_ph', self.orch_future_ph)
		if model.optimize():
			self.saver = tf.train.Saver()
		else:
			self.saver = None
		return

	def load_pretrained_model(self, path_to_model):
		# Restore model and preds graph
		self.saver = tf.train.import_meta_graph(path_to_model + '/model.meta')
		inputs_ph = tf.get_collection('inputs_ph')
		self.piano_t_ph, self.piano_past_ph, self.piano_future_ph, self.orch_past_ph, self.orch_future_ph = inputs_ph
		self.orch_t_ph = tf.get_collection("orch_t_ph")[0]
		self.preds = tf.get_collection("preds")[0]
		self.loss = tf.get_collection("loss")[0]
		self.loss_val = tf.get_collection("loss_val")[0]
		self.mask_orch_ph = tf.get_collection("mask_orch_ph")[0]
		self.train_step = tf.get_collection('train_step')[0]
		self.keras_learning_phase = tf.get_collection("keras_learning_phase")[0]
		return

	def build_feed_dict(self, batch_index, piano, orch, mask_orch):
		# Build batch
		piano_t, piano_past, piano_future, orch_past, orch_future, orch_t, mask_orch_t = build_batch(batch_index, piano, orch, mask_orch, len(batch_index), self.temporal_order)

		# Train step
		feed_dict = {self.piano_t_ph: piano_t,
			self.piano_past_ph: piano_past,
			self.piano_future_ph: piano_future,
			self.orch_past_ph: orch_past,
			self.orch_future_ph: orch_future,
			self.orch_t_ph: orch_t,
			self.mask_orch_ph: mask_orch_t}
		return feed_dict, orch_t

	def build_feed_dict_long_range(self, t, piano_extracted, orch_extracted, orch_gen):
		# We cannot use build_batch function here, but getting the matrices is quite easy
		piano_t = piano_extracted[:, t, :]
		piano_past = piano_extracted[:, t-(self.temporal_order-1):t, :]
		piano_future = piano_extracted[:, t+1:t+self.temporal_order, :]
		orch_t = orch_extracted[:, t, :]
		orch_past = orch_gen[:, t-(self.temporal_order-1):t, :]
		orch_future = orch_gen[:, t+1:t+self.temporal_order, :]
		mask_orch_t = np.ones_like(orch_t)
		
		# Train step
		feed_dict = {self.piano_t_ph: piano_t,
			self.piano_past_ph: piano_past,
			self.piano_future_ph: piano_future,
			self.orch_past_ph: orch_past,
			self.orch_future_ph: orch_future,
			self.orch_t_ph: orch_t,
			self.mask_orch_ph: mask_orch_t}
		return feed_dict, orch_t

	def training_step(self, sess, batch_index, piano, orch, mask_orch, summarize_dict):
		feed_dict, _ = self.build_feed_dict(batch_index, piano, orch, mask_orch)
		feed_dict[self.keras_learning_phase] = True

		SUMMARIZE = summarize_dict['bool']
		merged_node = summarize_dict['merged_node']

		#############################################
		#############################################
		#############################################
		# Compute test Jacobian, to check that gradients are set to zero : Test passed !
		# for trainable_parameter in tf.trainable_variables():
		# 	if trainable_parameter.name == "dense_3/bias:0":
		# 		AAA = trainable_parameter
		# grads = tf.gradients(self.loss, AAA)
		# loss_batch, dydx = sess.run([self.loss, grads], feed_dict)
		# import pdb; pdb.set_trace()
		#############################################
		#############################################
		#############################################
		
		if SUMMARIZE:
			_, loss_batch, preds_batch, sparse_loss_batch, summary = sess.run([self.train_step, self.loss, self.preds, self.sparse_loss_mean, merged_node], feed_dict)
		else:
			_, loss_batch, preds_batch, sparse_loss_batch = sess.run([self.train_step, self.loss, self.preds, self.sparse_loss_mean], feed_dict)
			summary = None

		debug_outputs = [sparse_loss_batch]

		return loss_batch, preds_batch, debug_outputs, summary

	def valid_step(self, sess, batch_index, piano, orch, mask_orch, PLOTING_FOLDER):
		# Almost the same function as training_step here,  but in the case of NADE learning for instance, it might be ver different.
		feed_dict, orch_t = self.build_feed_dict(batch_index, piano, orch, mask_orch)
		feed_dict[self.keras_learning_phase] = False
		loss_batch, preds_batch = sess.run([self.loss_val, self.preds], feed_dict)
		return loss_batch, preds_batch, orch_t

	def valid_long_range_step(self, sess, t, piano_extracted, orch_extracted, orch_gen):
		feed_dict, orch_t = self.build_feed_dict_long_range(t, piano_extracted, orch_extracted, orch_gen)
		feed_dict[self.keras_learning_phase] = False
		loss_batch, preds_batch = sess.run([self.loss_val, self.preds], feed_dict)
		return loss_batch, preds_batch, orch_t

	def generation_step(self, sess, batch_index, piano, orch_gen, mask_orch):
		# Exactly the same as the valid_step in the case of the standard_learner
		loss_batch, preds_batch, orch_t = self.valid_step(sess, batch_index, piano, orch_gen, mask_orch, None)
		return preds_batch