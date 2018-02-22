#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np
import keras
import random
import time
import config
from keras import backend as K

from LOP.Utils.build_batch import build_batch
from LOP.Scripts.standard_learning.standard_trainer import Standard_trainer

class NADE_trainer(Standard_trainer):
	
	def __init__(self, **kwargs):
		Standard_trainer.__init__(self, **kwargs)
		# Number of ordering used when bagging NADEs
		self.num_ordering = kwargs["num_ordering"]
		return

	def build_variables_nodes(self, model, parameters):
		Standard_trainer.build_variables_nodes(self, model, parameters)
		self.mask_input = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="mask_input")		
		self.orch_pred = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="orch_pred")		
		return

	def build_preds_nodes(self, model):
		inputs_ph = (self.piano_t_ph, self.piano_past_ph, self.piano_future_ph, self.orch_past_ph, self.orch_future_ph)
		# Prediction
		self.preds, self.embedding_concat = model.predict(inputs_ph, self.orch_pred, self.mask_input)
		return
	
	def build_loss_nodes(self, model, parameters):
		with tf.name_scope('loss'):
			with tf.name_scope('distance'):
				distance = Standard_trainer.build_distance(self, model, parameters)
		
			if model.sparsity_coeff != 0:
				with tf.name_scope('sparse_output_constraint'):
					sparse_loss, sparse_loss_mean = self.build_sparsity_term(model, parameters)
					loss_val_ = distance + sparse_loss
					self.sparse_loss_mean = sparse_loss_mean
			else:
				loss_val_ = distance
				temp = tf.zeros_like(loss_val_)
				self.sparse_loss_mean = tf.reduce_mean(temp)

			with tf.name_scope("NADE_mask_input"):
				##################################################
				##################################################
				# Masked gradients are the values known in the input : so 1 - mask are used for gradient 
				# loss_val_masked = tf.boolean_mask(self.mask_input==1, loss_val_, tf.zeros_like(loss_val_))
				loss_val_masked_ = tf.stop_gradient(self.mask_input*loss_val_) + (1-self.mask_input)*loss_val_
				# LEQUEL ??
				##################################################
				##################################################
				# Normalization
				norm_nade = model.orch_dim / (model.orch_dim - tf.reduce_sum(self.mask_input, axis=1) + 1)
				loss_val_masked = model.orch_dim

			# Note: don't reduce_mean the validation loss, we need to have the per-sample value
			if parameters['mask_orch']:
				with tf.name_scope('mask_orch'):
					loss_masked = tf.where(mask_orch_ph==1, loss_val_, tf.zeros_like(loss_val_))
					self.loss_val = loss_masked
			else:
				self.loss_val = loss_val_
		
			mean_loss = tf.reduce_mean(self.loss_val)
			with tf.name_scope('weight_decay'):
				weight_decay = Standard_trainer.build_weight_decay(self, model)
				# Weight decay
				if model.weight_decay_coeff != 0:
					# Keras weight decay does not work...
					self.loss = mean_loss + weight_decay
				else:
					self.loss = mean_loss
		return

	def build_train_step_node(self, model, optimizer):
		Standard_trainer.build_train_step_node(self, model, optimizer)
		return
	
	def save_nodes(self, model):
		Standard_trainer.save_nodes(self, model)
		tf.add_to_collection('mask_input', self.mask_input)
		tf.add_to_collection('orch_pred', self.orch_pred)
		return

	def load_pretrained_model(self, path_to_model):
		# Restore model and preds graph
		Standard_trainer.load_pretrained_model(self, path_to_model)
		self.mask_input = tf.get_collection('mask_input')[0]
		self.orch_pred = tf.get_collection('orch_pred')[0]
		return

	def training_step(self, sess, batch_index, piano, orch, mask_orch, summarize_dict):
		feed_dict, orch_t = Standard_trainer.build_feed_dict(self, batch_index, piano, orch, mask_orch)
		feed_dict[self.keras_learning_phase] = True
		
		# Generate a mask for the input
		batch_size, orch_dim = orch_t.shape
		mask = np.zeros_like(orch_t)
		for batch_ind in range(batch_size):
			# Number of known units
			d = random.randint(0, orch_dim)
			# Indices
			ind = np.random.random_integers(0, orch_dim-1, (d,))
			mask[batch_ind, ind] = 1
		
		feed_dict[self.mask_input] = mask
		# No need to mask orch_t here, its done in the tensorflow graph
		feed_dict[self.orch_pred] = orch_t

		SUMMARIZE = summarize_dict['bool']
		merged_node = summarize_dict['merged_node']
		
		if SUMMARIZE:
			_, loss_batch, preds_batch, sparse_loss_batch, summary = sess.run([self.train_step, self.loss, self.preds, self.sparse_loss_mean, merged_node], feed_dict)
		else:
			_, loss_batch, preds_batch, sparse_loss_batch = sess.run([self.train_step, self.loss, self.preds, self.sparse_loss_mean], feed_dict)
			summary = None

		debug_outputs = [sparse_loss_batch]

		return loss_batch, preds_batch, debug_outputs, summary

	def generate_mean_ordering(self, sess, feed_dict, orch_t):
		
		batch_size, orch_dim = orch_t.shape
		loss_batch_list = []

		# Generate the orderings in parallel -> duplicate the matrices along batch dim
		for k, v in feed_dict.iteritems():
			new_v = np.concatenate([v for _ in range(self.num_ordering)], axis=0)
			# DEBUG, can be removed later
			assert[np.all(new_v[:batch_size] == v)], "problem when duplicating v"
			feed_dict[k] = new_v
		
		# Also duplicate orch_t
		new_orch_t = np.concatenate([orch_t for _ in range(self.num_ordering)], axis=0)
		# Start with an orchestra prediction and mask equal to zero
		orch_pred = np.zeros_like(new_orch_t)
		mask = np.zeros_like(new_orch_t)

		# TODO (izy) : possibility to use a fixed precomputed orderings
		# Build the orderings (use the same ordering for all elems in batch)
		orderings = []
		for ordering_ind in range(self.num_ordering):
			# This ordering
			ordering = range(orch_dim)
			random.shuffle(ordering)
			orderings.append(ordering)

		# Loop over the length of the orderings
		for d in range(orch_dim):

			# Generate step
			feed_dict[self.orch_pred] = orch_pred
			feed_dict[self.mask_input] = mask
			
			loss_batch, preds_batch = sess.run([self.loss_val, self.preds], feed_dict)

			loss_batch_list.append(loss_batch)
			
			# Update matrices
			for ordering_ind in range(self.num_ordering):
				batch_begin = batch_size * ordering_ind
				batch_end = batch_size * (ordering_ind+1)
				mask[batch_begin:batch_end, orderings[ordering_ind][d]] = 1

				##################################################
				# Do we sample or not ??????
				orch_pred[batch_begin:batch_end, orderings[ordering_ind][d]] = np.random.binomial(1, preds_batch[batch_begin:batch_end, orderings[ordering_ind][d]])
				##################################################

		# Mean over the different generations (Comb filter output)
		preds_mean_over_ordering = np.zeros((batch_size, orch_dim))
		ind_orderings = np.asarray([e*batch_size for e in range(self.num_ordering)])
		for ind_batch in range(batch_size):
			preds_mean_over_ordering[ind_batch, :] = np.mean(orch_pred[ind_orderings, :], axis=0)
			ind_orderings += 1

		# Ca c'est n'importe quoi
		loss_batch_mean = np.mean(loss_batch_list)

		return loss_batch_mean, preds_mean_over_ordering

	def valid_step(self, sess, batch_index, piano, orch, mask_orch):
		feed_dict, orch_t = Standard_trainer.build_feed_dict(self, batch_index, piano, orch, mask_orch)
		loss_batch, preds_batch = self.generate_mean_ordering(sess, feed_dict, orch_t)
		return loss_batch, preds_batch, orch_t

	def valid_long_range_step(self, sess, t, piano_extracted, orch_extracted, orch_gen):
		# This takes way too much time in the case of NADE, so just remove it
		feed_dict, orch_t = Standard_trainer.build_feed_dict_long_range(self, t, piano_extracted, orch_extracted, orch_gen)
		# loss_batch, preds_batch  = self.generate_mean_ordering(sess, feed_dict, orch_t)
		loss_batch = 0.
		preds_batch = np.zeros_like(orch_t)
		return loss_batch, preds_batch, orch_t

	def generation_step(self, sess, batch_index, piano, orch_gen, mask_orch):
		# Exactly the same as the valid_step
		loss_batch, preds_batch, orch_t = self.valid_step(sess, batch_index, piano, orch_gen, mask_orch)
		return preds_batch