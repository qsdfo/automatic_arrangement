#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Simple feedforward MLP from the piano input with a binary cross-entropy cost
# Used to test main scripts and as a baseline

from LOP.Models.model_lop import Model_lop
from LOP.Models.Utils.weight_summary import variable_summary

# Tensorflow
import tensorflow as tf

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp


class Conv_piano(Model_lop):
	def __init__(self, model_param, dimensions):

		Model_lop.__init__(self, model_param, dimensions)

		# Stack conv
		self.filters = model_param["num_filter_piano"]
		self.kernels = model_param["kernel_size_piano"]

		return

	@staticmethod
	def name():
		return "Baseline_Conv_piano"
	@staticmethod
	def binarize_piano():
		return True
	@staticmethod
	def binarize_orchestra():
		return True
	@staticmethod
	def is_keras():
		return True
	@staticmethod
	def optimize():
		return True

	@staticmethod
	def get_hp_space():
		super_space = Model_lop.get_hp_space()

		space = {
			'filter_0': quniform_int('filter_0', 20, 50, 1),
			'kernel_0': quniform_int('kernel_0', 8, 16, 1),
			'filter_1': quniform_int('filter_1', 20, 50, 1),
			'kernel_1': quniform_int('kernel_1', 8, 16, 1),
		}

		space.update(super_space)
		return space

	def predict(self, piano_t, orch_past):
		#####################
		# Piano embedding
		with tf.name_scope("conv_piano"):
			conv_layer = Conv1D(self.num_filter_piano, self.kernel_size_piano, activation='relu')
			p0 = conv_layer(tf.reshape(piano_t, [-1, self.piano_dim, 1]))
			keras_layer_summary(conv_layer)
		
		import pdb; pdb.set_trace()



		with tf.name_scope("weighted_sum_piano"):
			W = tf.get_variable("W", shape=(self.num_filter_piano,))
			p1 = tf.scalar_mul(1 / tf.reduce_sum(W), tf.tensordot(p0, W, [[2],[0]]))
			variable_summary(W)
		piano_embedding = MLP(p1, self.mlp_piano, "mlp_piano", activation='relu')
		#####################

		#####################
		# GRU for modelling past orchestra
		# First layer
		if len(self.gru_orch) > 1:
			return_sequences = True
		else:
			return_sequences = False
		
		with tf.name_scope("orch_rnn_0"):
			x = GRU(self.gru_orch[0], return_sequences=return_sequences, input_shape=(self.temporal_order, self.orch_dim),
					activation='relu', dropout=self.dropout_probability,
					kernel_regularizer=keras.regularizers.l2(self.weight_decay_coeff),
					bias_regularizer=keras.regularizers.l2(self.weight_decay_coeff))(orch_past)
		
		if len(self.gru_orch) > 1:
			# Intermediates layers
			for layer_ind in range(1, len(self.gru_orch)):
				# Last layer ?
				if layer_ind == len(self.gru_orch)-1:
					return_sequences = False
				else:
					return_sequences = True
				with tf.name_scope("orch_rnn_" + str(layer_ind)):
					x = GRU(self.gru_orch[layer_ind], return_sequences=return_sequences,
							activation='relu', dropout=self.dropout_probability,
							kernel_regularizer=keras.regularizers.l2(self.weight_decay_coeff),
							bias_regularizer=keras.regularizers.l2(self.weight_decay_coeff))(x)

		orch_embedding = x
		#####################
		
		#####################
		# Prediction
		input_pred = tf.concat([piano_embedding, orch_embedding], axis=1)
		top_input = MLP(input_pred, self.mlp_pred, "mlp_pred", activation='relu')
		# Dense layers on top
		with tf.name_scope("last_MLP"):
			orch_prediction = Dense(self.orch_dim, activation='sigmoid', name='orch_pred')(top_input)
		#####################

		return orch_prediction
		return orch_prediction