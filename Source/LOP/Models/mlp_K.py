#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Keras version of the MLP :
# Simple feedforward MLP from the piano input with a binary cross-entropy cost
# Used to test main scripts and as a baseline

from LOP.Models.model_lop import Model_lop

# Tensorflow
import tensorflow as tf

# Keras
from keras.layers import Dense, Dropout

# Hyperopt
from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp


class MLP_K(Model_lop):
	def __init__(self, model_param, dimensions):

		Model_lop.__init__(self, model_param, dimensions)

		# Hidden layers architecture
		self.layers = model_param['layers']
		# Is it a keras model ?
		self.keras = False

		return

	@staticmethod
	def name():
		return "MLP_keras"

	@staticmethod
	def get_hp_space():
		super_space = Model_lop.get_hp_space()

		space = {'layers': hp.choice('layers', [
				[],
				[hopt_wrapper.qloguniform_int('layer_'+str(i), log(100), log(5000), 10) for i in range(1)],
				[hopt_wrapper.qloguniform_int('layer_'+str(i), log(100), log(5000), 10) for i in range(2)],
				[hopt_wrapper.qloguniform_int('layer_'+str(i), log(100), log(5000), 10) for i in range(3)],
			]),
		}

		space.update(super_space)
		return space

	def predict(self):
		x = self.piano_t
		
		for l in self.layers:
			x = Dense(l, activation='relu')(x)
			x = Dropout(0.5)(x)

		orch_prediction = Dense(self.orch_dim, activation='sigmoid')(x)

		return orch_prediction