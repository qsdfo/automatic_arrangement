#!/usr/bin/env python
# -*- coding: utf8 -*-

# Wrapper for stacked recurrent nets

import tensorflow as tf
import keras
from keras.layers.recurrent import GRU, LSTM
from LOP.Models.Utils.weight_summary import keras_layer_summary


def stacked_rnn(input_seq, layers, rnn_type='gru', weight_decay_coeff=0, dropout_probability=0, activation='relu'):

	def layer_rnn(layer, rnn_type, return_sequences):
		if rnn_type is 'gru':
			this_layer = GRU(layer, return_sequences=return_sequences,
					activation=activation, dropout=dropout_probability,
					kernel_regularizer=keras.regularizers.l2(weight_decay_coeff),
					bias_regularizer=keras.regularizers.l2(weight_decay_coeff))
		return this_layer

	if len(layers) > 1:
		return_sequences = True
	else:
		return_sequences = False

	with tf.name_scope("0"):
		this_layer = layer_rnn(layers[0], rnn_type, return_sequences)
		x = this_layer(input_seq)
		keras_layer_summary(this_layer)
	
	if len(layers) > 1:
		# Intermediates layers
		for layer_ind in range(1, len(layers)):
			# Last layer ?
			if layer_ind == len(layers)-1:
				return_sequences = False
			else:
				return_sequences = True
			with tf.name_scope(str(layer_ind)):
				this_layer = layer_rnn(layers[layer_ind], rnn_type, return_sequences)
				x = this_layer(x)
				keras_layer_summary(this_layer)

	return x