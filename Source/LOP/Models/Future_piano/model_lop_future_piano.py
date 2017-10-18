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

from LOP.Models.Utils.weight_summary import keras_layer_summary


class MLFP(Model_lop):
	
	def __init__(self, model_param, dimensions):
		Model_lop.__init__(self, model_param, dimensions)
		return

	@staticmethod
	def name():
		return "Future_piano_"