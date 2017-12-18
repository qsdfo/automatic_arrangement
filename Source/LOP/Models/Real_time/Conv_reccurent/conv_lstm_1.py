#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop

# Tensorflow
import tensorflow as tf

# Keras
import keras
from keras.layers import Conv1D, GRU, Dense

# Hyperopt
from LOP.Utils.hopt_wrapper import quniform_int, qloguniform_int
from LOP.Utils.hopt_utils import list_log_hopt, list_hopt_fixedSized
from math import log

from LOP.Models.Utils.mlp import MLP
from LOP.Models.Utils.weight_summary import variable_summary, keras_layer_summary


class Conv_lstm_1(Model_lop):
	def __init__(self, model_param, dimensions):
		Model_lop.__init__(self, model_param, dimensions)
		self.num_filter_piano = model_param["num_filter_piano"]
		self.kernel_size_piano = model_param["kernel_size_piano"]
		self.num_filter_orch = model_param["num_filter_orch"]
		self.kernel_size_orch = model_param["kernel_size_orch"]
		self.embeddings_size = model_param["embeddings_size"]
		# The last recurrent layer output a vector of dimension embedding size
		self.gru_orch = list(model_param["gru_orch"])
		self.gru_orch.append(self.embeddings_size)
		self.mlp_pred = model_param["mlp_pred"]
		return

	@staticmethod
	def name():
		return "Conv_lstm_1"
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
			'num_filter_piano': list_hopt_fixedSized([(20,30,1), (10,20,1)], 'num_filter_piano'),
			'kernel_size_piano': list_hopt_fixedSized([(12, 24, 1), (12, 24, 1)], "kernel_size_piano"),
			'num_filter_orch': list_hopt_fixedSized([(30,50,1), (10,20,1)], 'num_filter_orch'),
			'kernel_size_orch': list_hopt_fixedSized([(12, 24, 1), (12, 24, 1)], "kernel_size_orch"),
			'embeddings_size': qloguniform_int("embeddings_size", log(500), log(2000), 10),
			'mlp_pred': list_log_hopt(500, 2000, 10, 1, 3, "mlp_pred"),
			'gru_orch': list_log_hopt(500, 2000, 10, 0, 2, "gru_orch"),
		}
		space.update(super_space)
		return space

	def predict(self, inputs_ph):
		
		piano_t, _, _, orch_past, _ = inputs_ph

		#####################
		# Piano embedding
		with tf.name_scope("piano_embedding"):
			with tf.name_scope("conv_0"):
				conv_layer = Conv1D(self.num_filter_piano[0], self.kernel_size_piano[0], activation='relu')
				piano_t_reshape = tf.reshape(piano_t, [-1, self.piano_dim, 1])
				_P0_ = conv_layer(piano_t_reshape)
				keras_layer_summary(conv_layer)

			# Second level convolution :
			# conv along pitch axis = (num_pitch - kernel[0])
			# So each kernel takes as input a matrix (pitch-kernel[0]+1, num_filter[0]) and outputs a scalar
			# Then, output has size (pitch-kernel[0]-kernel[1]+2, num_filter[1])
			with tf.name_scope("conv_1"):
				conv_layer = Conv1D(self.num_filter_piano[1], self.kernel_size_piano[1], activation='relu')
				_P1_ = conv_layer(_P0_)
				keras_layer_summary(conv_layer)
			
			dims_last_layer = _P1_.shape.as_list()
			_P2_ = tf.reshape(_P1_, [-1,  dims_last_layer[1]* dims_last_layer[2]])
			
			piano_embedding = MLP(_P2_, [self.embeddings_size], "adapt_size", activation='relu')
		#####################

		#####################
		# Convolutions over pitch axis
		with tf.name_scope("embedding_orch"):
			with tf.name_scope("pitch_convolution"):
				_O0_ = tf.reshape(orch_past, [-1, self.orch_dim])
				with tf.name_scope("0"):
					conv_layer = Conv1D(self.num_filter_orch[0], self.kernel_size_orch[0], activation='relu')
					_O1_ = conv_layer(tf.reshape(_O0_, [-1, self.orch_dim, 1]))
					keras_layer_summary(conv_layer)

				with tf.name_scope("1"):
					conv_layer = Conv1D(self.num_filter_orch[1], self.kernel_size_orch[1], activation='relu')
					_O2_ = conv_layer(_O1_)
					keras_layer_summary(conv_layer)
				
				dims_last_layer = _O2_.shape.as_list()
				dim_last_layer = dims_last_layer[1]* dims_last_layer[2]
				# Reshape into (batch, time, features)
				_O3_ = tf.reshape(_O2_, [-1, self.temporal_order, dim_last_layer])

			with tf.name_scope("time_recurrence"):
				# Recurrence over time axis
				if len(self.gru_orch) > 1:
					return_sequences = True
				else:
					return_sequences = False
				
				with tf.name_scope("0"):
					x = GRU(self.gru_orch[0], return_sequences=return_sequences, input_shape=(self.temporal_order, dim_last_layer),
							activation='relu', dropout=self.dropout_probability)(orch_past)
				
				if len(self.gru_orch) > 1:
					# Intermediates layers
					for layer_ind in range(1, len(self.gru_orch)):
						# Last layer ?
						if layer_ind == len(self.gru_orch)-1:
							return_sequences = False
						else:
							return_sequences = True
						with tf.name_scope(str(layer_ind)):
							x = GRU(self.gru_orch[layer_ind], return_sequences=return_sequences,
									activation='relu', dropout=self.dropout_probability)(x)
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


# "0" : {
# 'temporal_order': 4, 
# 'weight_decay_coeff': 0.0001, 
# 'gru_orch': (500,), 
# 'num_filter_piano': (30, 20),
# 'kernel_size_piano': (12, 12),
# 'num_filter_orch': (40, 30),
# 'kernel_size_orch': (12, 12), 
# 'batch_size': 100, 
# 'dropout_probability': 0.2
# 'elbedding_size': 500
# }