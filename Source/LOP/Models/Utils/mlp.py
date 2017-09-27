#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
from keras.layers import Dense
from LOP.Models.Utils.weight_summary import keras_layer_summary

def MLP(x, layers, scope_name, activation='relu'):
	for layer_ind, num_unit in enumerate(layers):
		with tf.variable_scope(scope_name + "_" + str(layer_ind)):
			dense_layer = Dense(num_unit, activation=activation)
			x = dense_layer(x)
			keras_layer_summary(dense_layer)
	return x
