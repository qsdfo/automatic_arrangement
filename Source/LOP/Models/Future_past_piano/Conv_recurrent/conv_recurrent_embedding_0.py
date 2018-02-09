#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.Future_past_piano.model_lop_future_past_piano import MLFPP

# Tensorflow
import tensorflow as tf

# Keras
import keras
from keras.layers import Dense, Conv1D, TimeDistributed

# Hyperopt
from math import log

from LOP.Models.Utils.weight_summary import keras_layer_summary
from LOP.Models.Utils.stacked_rnn import stacked_rnn
from LOP.Utils.hopt_utils import list_log_hopt
from LOP.Utils.hopt_wrapper import qloguniform_int, quniform_int

class Conv_recurrent_embedding_0(MLFPP):
    """Recurrent embeddings for both the piano and orchestral scores
    Piano embedding : p(t), ..., p(t+N) through a convulotional layer and a stacked RNN. Last time index of last layer is taken as embedding.
    Orchestra embedding : o(t-N), ..., p(t) Same architecture than piano embedding.
    Then, the concatenation of both embeddings is passed through a MLP
    """
    def __init__(self, model_param, dimensions):

        MLFPP.__init__(self, model_param, dimensions)

        # Piano embedding
        self.kernel_size_piano = model_param["kernel_size_piano"] # only pitch_dim, add 1 for temporal conv 
        embeddings_size = model_param['embeddings_size']
        temp = model_param['hs_piano']
        self.hs_piano = list(temp)
        self.hs_piano.append(embeddings_size)

        # Orchestra embedding
        self.kernel_size_orch = model_param["kernel_size_orch"] # only pitch_dim, add 1 for temporal conv 
        temp = model_param['hs_orch']
        self.hs_orch = list(temp)
        self.hs_orch.append(embeddings_size)

        return

    @staticmethod
    def name():
        return (MLFPP.name() + "Recurrent_embeddings_0")
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
        super_space = MLFPP.get_hp_space()

        space = {
            'kernel_size_piano': quniform_int('kernel_size_piano', 4, 24, 1),
            'kernel_size_orch': quniform_int('kernel_size_orch', 4, 24, 1),
            'hs_piano': list_log_hopt(500, 2000, 10, 0, 2, 'hs_piano'),
            'hs_orch': list_log_hopt(500, 2000, 10, 0, 2, 'hs_orch'),
            'embeddings_size': qloguniform_int('hs_orch', log(500), log(1000), 10),
        }

        space.update(super_space)
        return space

    def piano_embedding(self, piano_t, piano_seq, reverse):
        with tf.name_scope("build_piano_input"):
            # Add a time axis to piano_t
            piano_t_time = tf.reshape(piano_t, [-1, 1, self.piano_dim])
            # Concatenate t and future
            input_seq = tf.concat([piano_t_time, piano_seq], 1)
            if reverse:
                # Flip the matrix along the time axis so that the last time index is t
                input_seq = tf.reverse(input_seq, [1])
            # Format as a 4D
            input_seq = tf.reshape(input_seq, [-1, self.temporal_order, self.piano_dim, 1])

        with tf.name_scope("conv_piano"):
            conv_layer = Conv1D(1, self.kernel_size_piano, activation='relu')
            conv_layer_timeDist = TimeDistributed(conv_layer, input_shape=(self.temporal_order, self.piano_dim, 1))
            p0 = conv_layer_timeDist(input_seq)
            keras_layer_summary(conv_layer)

        # Remove the last useless dimension
        cropped_piano_dim = self.piano_dim-self.kernel_size_piano+1
        p0 = tf.reshape(p0, [-1, self.temporal_order, cropped_piano_dim])

        with tf.name_scope("gru"):
            piano_emb = stacked_rnn(p0, self.hs_piano, 
                rnn_type='gru', 
                weight_decay_coeff=self.weight_decay_coeff,
                dropout_probability=self.dropout_probability, 
                activation='relu'
                )
        return piano_emb

    def orchestra_embedding(self, orch_past):
        with tf.name_scope("build_orch_input"):
            # Format as a 4D
            input_seq = tf.reshape(orch_past, [-1, self.temporal_order-1, self.orch_dim, 1])

        with tf.name_scope("conv_orch"):
            conv_layer = Conv1D(1, self.kernel_size_orch, activation='relu')
            conv_layer_timeDist = TimeDistributed(conv_layer, input_shape=(self.temporal_order-1, self.orch_dim, 1))
            o0 = conv_layer_timeDist(input_seq)
            keras_layer_summary(conv_layer)

        # Remove the last useless dimension
        cropped_orch_dim = self.orch_dim-self.kernel_size_orch+1
        o0 = tf.reshape(o0, [-1, self.temporal_order-1, cropped_orch_dim])
        
        with tf.name_scope("gru"):
            orchestra_embedding = stacked_rnn(o0, self.hs_orch, 
                rnn_type='gru', 
                weight_decay_coeff=self.weight_decay_coeff, 
                dropout_probability=self.dropout_probability, 
                activation='relu'
                )       
        return orchestra_embedding

    def predict(self, inputs_ph):

        piano_t, piano_past, piano_future, orch_past, _ = inputs_ph
        
        with tf.name_scope("piano_embedding_past"):
            piano_embedding_past = self.piano_embedding(piano_t, piano_past, reverse=False)

        with tf.name_scope("piano_embedding_future"):
            piano_embedding_future = self.piano_embedding(piano_t, piano_future, reverse=True)

        with tf.name_scope("orchestra_embedding"):
            orchestra_embedding = self.orchestra_embedding(orch_past)

        #####################
        # Concatenate and predict
        with tf.name_scope("top_layer_prediction_0"):
            top_input = keras.layers.concatenate([orchestra_embedding, piano_embedding_past, piano_embedding_future], axis=1)
            dense_layer = Dense(1000, activation='relu', name='orch_pred_0')
            top_0 = dense_layer(top_input)
            keras_layer_summary(dense_layer)
        with tf.name_scope("top_layer_prediction_1"):
            dense_layer = Dense(self.orch_dim, activation='sigmoid', name='orch_pred')
            orch_prediction = dense_layer(top_0)
            keras_layer_summary(dense_layer)
        #####################

        return orch_prediction, top_input


# 'temporal_order' : 5,
# 'dropout_probability' : 0,
# 'weight_decay_coeff' : 0,
# 'kernel_size_piano': 12,
# 'kernel_size_orch': 12,
# 'hs_piano': [500],
# 'hs_orch': [600],
# 'embeddings_size': 500,