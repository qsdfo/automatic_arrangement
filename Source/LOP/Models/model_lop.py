#!/usr/bin/env python
# -*- coding: utf8 -*-


from abc import ABCMeta, abstractmethod


class Model_lop(object):
	
	__metaclass__ = ABCMeta

	def __init__(model_param, dimensions):
		# Dimensions
		self.batch_size = dimensions['batch_size']
        self.temporal_order = dimensions['temporal_order']
        self.piano_dim = dimensions['piano_dim']
        self.orchestra_dim = dimensions['orchestra_dim']

        # Regularization paramters
        self.dropout_probability = model_param['dropout_probability']
        self.weight_decay_coeff = model_param['weight_decay_coeff']

        # Numpy and theano random generators
        self.rng_np = RandomState(25)
        self.rng = RandomStreams(seed=25)

        self.params = []
		return

	@abstractmethod
	def predict():
		pass

	@abstractmethod
	def train_step():
		pass

	@abstractmethod
	def validate():
		pass

	@abstractmethod
	def name():
		pass

	@staticmethod
	def get_hp_space():
		space_training = {'batch_size': hopt_wrapper.quniform_int('batch_size', 50, 500, 1),
                          'temporal_order': hopt_wrapper.qloguniform_int('temporal_order', log(3), log(20), 1)
                          }

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