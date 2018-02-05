#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Keras version of the MLP :
# Simple feedforward MLP from the piano input with a binary cross-entropy cost
# Used to test main scripts and as a baseline

from LOP.Models.model_lop import Model_lop
from LOP.Utils.hopt_utils import list_log_hopt
from LOP.Models.Utils.weight_summary import keras_layer_summary

# Tensorflow
import tensorflow as tf

# Keras
from keras import regularizers
from keras.layers import Dense, Dropout

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp


class MLP_K(Model_lop):
	def __init__(self, model_param, dimensions):

		Model_lop.__init__(self, model_param, dimensions)

		# Hidden layers architecture
		self.layers = model_param['n_hidden']
		# Is it a keras model ?
		self.keras = True

		return

	@staticmethod
	def name():
		return "Baseline_MLP_keras"
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

		space = {'n_hidden': list_log_hopt(500, 2000, 10, 1, 2, "n_hidden")}

		space.update(super_space)
		return space

	def predict(self, inputs_ph):
		piano_t, _, _, orch_past, _ = inputs_ph
		
		# x = piano_t
		orch_past_flat = tf.reshape(orch_past, [-1, (self.temporal_order-1) * self.orch_dim])
		x = tf.concat([piano_t, orch_past_flat], axis=1)

		for i, l in enumerate(self.layers):
			with tf.name_scope("layer_" + str(i)):
				dense = Dense(l, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay_coeff))
				x = dense(x)
				keras_layer_summary(dense)
				x = Dropout(self.dropout_probability)(x)

		orch_prediction = Dense(self.orch_dim, activation='sigmoid')(x)

		return orch_prediction, None