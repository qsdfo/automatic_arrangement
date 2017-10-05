#!/usr/bin/env python
# -*- coding: utf8 -*-


import numpy as np
import cPickle as pkl
import tensorflow as tf
from keras import backend as K

from LOP.Utils.build_batch import build_batch


def generate(piano, path_to_config, orch_init=None, batch_size=5):
	# Perform N=batch_size orchestrations
	# Sample by sample generation
	# Input : 
	# 	- piano score : numpy (time, pitch)
	#  	- model
	# 	- optionnaly the beginning of an orchestral score : numpy (time, pitch)
	# Output :
	# 	- orchestration by the model	

	# Paths
	path_to_model = path_to_config + '/model'
	dimensions = pkl.load(open(path_to_config + '/dimensions.pkl', 'rb'))
	is_keras = pkl.load(open(path_to_config + '/is_keras.pkl', 'rb'))

	# Get dimensions
	piano_dim = dimensions['piano_dim']
	orch_dim = dimensions['orch_dim']
	temporal_order = dimensions['temporal_order']
	total_length = piano.shape[0]

	if orch_init is not None:
		init_length = orch_init.shape[0]
		assert (init_length < total_length), "Orchestration initialization is longer than the piano score"
		assert (init_length + 1 >= temporal_order), "Orchestration initialization must be longer than the temporal order of the model"
	else:
		init_length = temporal_order - 1
		orch_init = np.zeros((init_length, orch_dim))

	# Instanciate generation
	orch_gen = np.zeros((batch_size, total_length, orch_dim))
	orch_gen[:, :init_length, :] = orch_init

	# Restore model and preds graph
	tf.reset_default_graph() # First clear graph to avoid memory overflow when running training and generation in the same process
	saver = tf.train.import_meta_graph(path_to_model + '/model.meta')
	preds = tf.get_collection("preds")[0]
	keras_learning_phase = tf.get_collection("keras_learning_phase")[0]

	with tf.Session() as sess:
			
		if is_keras:
			K.set_session(sess)

		saver.restore(sess, path_to_model + '/model')

		# for t in range(init_length, total_length): 	A REMPLACER PAR TOTAL_LENGTH - TEMPORAL ORDER QUAND ON FERA AUSSI BACKAARD
		for t in range(init_length, total_length):
			# Just duplicate the temporal index to create batch generation
			batch_index = np.tile(t, batch_size)

			piano_t, orch_past, _ = build_batch(batch_index, piano, orch_gen, batch_size, temporal_order)

			# Feed dict
			feed_dict = {'piano_t:0': piano_t,
						'orch_past:0': orch_past,
						keras_learning_phase: 0}

			# Get prediction
			prediction = sess.run(preds, feed_dict)

			# Preds should be a probability distribution. Sample from it
			# Note that it doesn't need to be part of the graph since we don't use the sampled value to compute the backproped error 
			prediction_sampled = np.random.binomial(1, prediction)

			orch_gen[:, t, :] = prediction_sampled

	return orch_gen