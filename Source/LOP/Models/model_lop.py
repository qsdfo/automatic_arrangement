#!/usr/bin/env python
# -*- coding: utf8 -*-


from LOP.Utils import hopt_wrapper
from math import log
from hyperopt import hp


class Model_lop(object):
	
	def __init__(self, model_param, dimensions):
		# Dimensions
		self.temporal_order = dimensions['temporal_order']
		self.piano_dim = dimensions['piano_dim']
		self.orch_dim = dimensions['orch_dim']

		# Regularization paramters
		self.dropout_probability = model_param['dropout_probability']
		self.weight_decay_coeff = model_param['weight_decay_coeff']

		self.params = []
		return

	@staticmethod
	def get_hp_space():
		space_training = {'temporal_order': hopt_wrapper.qloguniform_int('temporal_order', log(3), log(20), 1)}

		space_regularization = {'dropout_probability': hp.choice('dropout', [
			0.0,
			hp.normal('dropout_probability', 0.5, 0.1)
		]),
			'weight_decay_coeff': hp.choice('weight_decay_coeff', [
				0.0,
				hp.uniform('a', 1e-4, 1e-4)
			]),
		}

		space_training.update(space_regularization)
		return space_training