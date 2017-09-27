#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop

# Tensorflow
import tensorflow as tf

# Keras
import keras
from keras.layers.recurrent import GRU
from keras.layers import Dense

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp


class LSTM_plugged_base(Model_lop):
	def __init__(self, model_param, dimensions):

		Model_lop.__init__(self, model_param, dimensions)

		# Hidden layers architecture
		self.n_hs = model_param['n_hidden']

		# Is it a keras model ?
		self.keras = True

		return

	@staticmethod
	def name():
		return "LSTM_plugged_base"
	@staticmethod
	def binarize_piano():
		return True
	@staticmethod
	def binarize_orchestra():
		return True

	@staticmethod
	def get_hp_space():
		super_space = Model_lop.get_hp_space()

		space = {'n_hidden': hp.choice('n_hidden', [
				[],
				[hopt_wrapper.qloguniform_int('n_hidden_1_'+str(i), log(100), log(5000), 10) for i in range(1)],
				[hopt_wrapper.qloguniform_int('n_hidden_2_'+str(i), log(100), log(5000), 10) for i in range(2)],
				[hopt_wrapper.qloguniform_int('n_hidden_3_'+str(i), log(100), log(5000), 10) for i in range(3)],
			]),
		}

		space.update(super_space)
		return space

	def predict(self, piano_t, orch_past):
		#####################
		# Batch norm
		# if self.binarize_piano:
		# 	piano_t = BatchNorm()
		#####################


		#####################
		# GRU for modelling past orchestra
		# First layer
		if len(self.n_hs) > 1:
			return_sequences = True
		else:
			return_sequences = False
		
		with tf.name_scope("orch_rnn_0"):
			x = GRU(self.n_hs[0], return_sequences=return_sequences, input_shape=(self.temporal_order, self.orch_dim),
					activation='relu', dropout=self.dropout_probability,
					kernel_regularizer=keras.regularizers.l2(self.weight_decay_coeff),
					bias_regularizer=keras.regularizers.l2(self.weight_decay_coeff))(orch_past)
		
		if len(self.n_hs) > 1:
			# Intermediates layers
			for layer_ind in range(1, len(self.n_hs)):
				# Last layer ?
				if layer_ind == len(self.n_hs)-1:
					return_sequences = False
				else:
					return_sequences = True
				with tf.name_scope("orch_rnn_" + str(layer_ind)):
					x = GRU(self.n_hs[layer_ind], return_sequences=return_sequences,
							activation='relu', dropout=self.dropout_probability,
							kernel_regularizer=keras.regularizers.l2(self.weight_decay_coeff),
							bias_regularizer=keras.regularizers.l2(self.weight_decay_coeff))(x)

		lstm_out = x
		#####################
		
		#####################
		# gru out and piano(t)
		with tf.name_scope("piano_embedding"):
			piano_embedding = Dense(self.n_hs[-1], activation='relu')(piano_t)  # fully-connected layer with 128 units and ReLU activation
		#####################

		#####################
		# Concatenate and predict
		with tf.name_scope("concatenation"):
			top_input = keras.layers.concatenate([lstm_out, piano_embedding], axis=1)
		# Dense layers on top
		orch_prediction = Dense(self.orch_dim, activation='sigmoid', name='orch_pred',
								kernel_regularizer=keras.regularizers.l2(self.weight_decay_coeff),
								bias_regularizer=keras.regularizers.l2(self.weight_decay_coeff))(top_input)
		#####################

		return orch_prediction