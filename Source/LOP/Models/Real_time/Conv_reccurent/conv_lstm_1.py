#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop

# Tensorflow
import tensorflow as tf

# Keras
import keras
from keras.layers import Conv1D, GRU, Dense

# Hyperopt
from LOP.Utils.hopt_wrapper import quniform_int
from LOP.Utils.hopt_utils import multi_layer_hopt

from LOP.Models.Utils.mlp import MLP
from LOP.Models.Utils.weight_summary import variable_summary, keras_layer_summary


class Conv_lstm_0(Model_lop):
	def __init__(self, model_param, dimensions):

		Model_lop.__init__(self, model_param, dimensions)

		self.num_filter_piano = model_param["num_filter_piano"]
		self.kernel_size_piano = model_param["kernel_size_piano"]
		self.mlp_piano = model_param["mlp_piano"]
		self.mlp_pred = model_param["mlp_pred"]
		self.gru_orch = model_param["gru_orch"]

		return

	@staticmethod
	def name():
		return "Conv_lstm_0"

	@staticmethod
	def get_hp_space():
		super_space = Model_lop.get_hp_space()

		space = {
			'num_filter_piano': quniform_int('num_filter_piano', 20, 50, 1),
			'kernel_size_piano': quniform_int('kernel_size_piano', 8, 16, 1),
			'mlp_piano': multi_layer_hopt(500, 2000, 10, 1, 3, "mlp_piano"),
			'mlp_pred': multi_layer_hopt(500, 2000, 10, 1, 3, "mlp_pred"),
			'gru_orch': multi_layer_hopt(500, 2000, 10, 1, 3, "gru_orch"),
		}
		space.update(super_space)
		return space

	def predict(self, piano_t, orch_past):

		#####################
		# Piano embedding
		with tf.name_scope("conv_piano_0"):
			conv_layer = Conv1D(self.num_filter_piano, self.kernel_size_piano, activation='relu')
			p0 = conv_layer(tf.reshape(piano_t, [-1, self.piano_dim, 1]))
			keras_layer_summary(conv_layer)
		with tf.name_scope("conv_piano_1"):
			conv_layer = Conv1D(self.num_filter_piano, self.kernel_size_piano, activation='relu')
			p1 = conv_layer(p0)
			keras_layer_summary(conv_layer)
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


# 'batch_size' : 200,
# 'temporal_order' : 5,
# 'dropout_probability' : 0,
# 'weight_decay_coeff' : 0,
# 'num_filter_piano': 20,
# 'kernel_size_piano': 12,
# 'mlp_piano': [500, 500],
# 'mlp_pred': [500, 500],
# 'gru_orch': [500, 500],