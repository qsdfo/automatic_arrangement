#!/usr/bin/env python
# -*- coding: utf8 -*-

# Model lop
from acidano.models.lop.model_lop import Model_lop

# Hyperopt
from hyperopt import hp
from acidano.utils import hopt_wrapper
from math import log

# Numpy
import numpy as np

# Theano
import theano
import theano.tensor as T

# Propagation
from acidano.utils.forward import propup_sigmoid, propup_tanh
# Performance measures
from acidano.utils.init import shared_normal, shared_zeros
from acidano.utils.measure import accuracy_measure, precision_measure, recall_measure
# Regularization
from acidano.utils.regularization import dropout_function
# Inputs
import acidano.utils.build_theano_input as build_theano_input


class LSTM(Model_lop):
    """LSTM multiple layers with regularization.

    Predictive model,
        input = piano(t)
        output = orchestra(t)
        measure = cross-entropy error function
            (output units are binary units (y_j) considered independent : i != j -> y_j indep y_i)

    With this model, silences are not automatically mapped to silences (just a problem of me being lazy).
    """

    def __init__(self,
                 model_param,
                 dimensions,
                 checksum_database,
                 weights_initialization=None):

        super(LSTM, self).__init__(model_param, dimensions, checksum_database)

        # Datas are represented like this:
        #   - visible = concatenation of the data : (num_batch, piano ^ orchestra_dim * temporal_order)
        self.n_v = dimensions['piano_dim']
        self.n_o = dimensions['orchestra_dim']

        # Number of hidden units
        self.n_hs = model_param['n_hidden']
        self.n_layer = len(self.n_hs)

        self.L_vi = {}
        self.L_hi = {}
        self.b_i = {}
        self.L_vc = {}
        self.L_hc = {}
        self.b_c = {}
        self.L_vf = {}
        self.L_hf = {}
        self.b_f = {}
        self.L_vo = {}
        self.L_ho = {}
        self.b_o = {}

        if weights_initialization is None:
            # Weights
            for layer in xrange(self.n_layer):
                if layer == 0:
                    n_htm1 = self.n_v
                else:
                    n_htm1 = self.n_hs[layer-1]
                n_ht = self.n_hs[layer]
                # input gate
                self.L_vi[layer] = shared_normal((n_htm1, n_ht), 0.01, name='L_vi'+str(layer))
                self.L_hi[layer] = shared_normal((n_ht, n_ht), 0.01, name='L_vi'+str(layer))
                self.b_i[layer] = shared_zeros((n_ht), name='b_i'+str(layer))
                # Internal cell
                self.L_vc[layer] = shared_normal((n_htm1, n_ht), 0.01, name='L_vc'+str(layer))
                self.L_hc[layer] = shared_normal((n_ht, n_ht), 0.01, name='L_hc'+str(layer))
                self.b_c[layer] = shared_zeros((n_ht), name='b_c'+str(layer))
                # Forget gate
                self.L_vf[layer] = shared_normal((n_htm1, n_ht), 0.01, name='L_vf'+str(layer))
                self.L_hf[layer] = shared_normal((n_ht, n_ht), 0.01, name='L_hf'+str(layer))
                self.b_f[layer] = shared_zeros((n_ht), name='b_f'+str(layer))
                # Output
                # No L_co... as in Theano tuto
                self.L_vo[layer] = shared_normal((n_htm1, n_ht), 0.01, name='L_vo'+str(layer))
                self.L_ho[layer] = shared_normal((n_ht, n_ht), 0.01, name='L_ho'+str(layer))
                self.b_o[layer] = shared_zeros((n_ht), name='b_o'+str(layer))

            # Last predictive layer
            self.W = shared_normal((self.n_hs[-1], self.n_o), 0.01, name='W')
            self.b = shared_zeros((self.n_o), name='b')
        else:
            # Layer weights
            for layer, n_h_layer in enumerate(self.n_hs):
                self.L_vi[layer] = weights_initialization['L_vi'][layer]
                self.L_hi[layer] = weights_initialization['L_hi'][layer]
                self.b_i[layer] = weights_initialization['b_i'][layer]
                self.L_vc[layer] = weights_initialization['L_vc'][layer]
                self.L_hc[layer] = weights_initialization['L_hc'][layer]
                self.b_c[layer] = weights_initialization['b_c'][layer]
                self.L_vf[layer] = weights_initialization['L_vf'][layer]
                self.L_hf[layer] = weights_initialization['L_hf'][layer]
                self.b_f[layer] = weights_initialization['b_f'][layer]
                self.L_vo[layer] = weights_initialization['L_vo'][layer]
                self.L_ho[layer] = weights_initialization['L_ho'][layer]
                self.b_o[layer] = weights_initialization['b_ho'][layer]
            self.W = weights_initialization['W']
            self.b = weights_initialization['b']

        self.params = self.L_vi.values() + self.L_hi.values() + self.b_i.values() + self.L_vc.values() + self.L_hc.values() +\
            self.b_c.values() + self.L_vf.values() + self.L_hf.values() + self.b_f.values() + self.L_vo.values() +\
            self.L_ho.values() + self.b_o.values() + [self.W, self.b]

        # Variables
        self.v = T.tensor3('v', dtype=theano.config.floatX)
        self.o = T.tensor3('o', dtype=theano.config.floatX)
        self.o_truth = T.tensor3('o_truth', dtype=theano.config.floatX)
        self.v_gen = T.tensor3('v_gen', dtype=theano.config.floatX)

        # Test values
        self.v.tag.test_value = np.random.rand(self.batch_size, self.temporal_order, self.n_v).astype(theano.config.floatX)
        self.o.tag.test_value = np.random.rand(self.batch_size, self.temporal_order, self.n_o).astype(theano.config.floatX)
        self.o_truth.tag.test_value = np.random.rand(self.batch_size, self.temporal_order, self.n_o).astype(theano.config.floatX)
        return

    ###############################
    #       STATIC METHODS
    #       FOR METADATA AND HPARAMS
    ###############################
    @staticmethod
    def get_hp_space():

        super_space = Model_lop.get_hp_space()

        space = {'n_hidden': hp.choice('n_hidden', [
            [hopt_wrapper.qloguniform_int('n_hidden_1_'+str(i), log(100), log(5000), 10) for i in range(1)],
            [hopt_wrapper.qloguniform_int('n_hidden_2_'+str(i), log(100), log(5000), 10) for i in range(2)],
            [hopt_wrapper.qloguniform_int('n_hidden_3_'+str(i), log(100), log(5000), 10) for i in range(3)],
            [hopt_wrapper.qloguniform_int('n_hidden_4_'+str(i), log(100), log(5000), 10) for i in range(4)]
        ]),
        }

        space.update(super_space)
        return space

    @staticmethod
    def name():
        return "LSTM"

    ###############################
    #  FORWARD PASS
    ###############################
    def corruption(self, h_lm1_t, n_lm1, axis):
        # Get input dimension (dieu que c'est moche)
        size_mask = (self.batch_size, n_lm1)
        # Apply dropout (see https://arxiv.org/pdf/1409.2329.pdf for details)
        # using a mask of zero
        if self.step_flag == 'train':
            h_lm1_t_corrupted = dropout_function(h_lm1_t, p_dropout=self.dropout_probability, size=size_mask, rng=self.rng)
        elif self.step_flag in ['validate', 'generate']:
            # Just multiply weights by the dropout ratio
            h_lm1_t_corrupted = h_lm1_t * (1-self.dropout_probability)
        else:
            raise ValueError("step_flag undefined")
        return h_lm1_t_corrupted

    def iteration(self, h_lm1_t, c_tm1, h_tm1,
                  L_vi, L_hi, b_i,
                  L_vf, L_hf, b_f,
                  L_vc, L_hc, b_c,
                  L_vo, L_ho, b_o,
                  n_lm1):
        # Sum along last axis
        axis = h_lm1_t.ndim - 1
        # Dropout
        h_lm1_t_corrupted = self.corruption(h_lm1_t, n_lm1, axis)
        # Input gate
        i = propup_sigmoid(T.concatenate([h_lm1_t_corrupted, h_tm1], axis=axis), T.concatenate([L_vi, L_hi]), b_i)
        # Forget gate
        f = propup_sigmoid(T.concatenate([h_lm1_t_corrupted, h_tm1], axis=axis), T.concatenate([L_vf, L_hf]), b_f)
        # Cell update term
        c_tilde = propup_tanh(T.concatenate([h_lm1_t_corrupted, h_tm1], axis=axis), T.concatenate([L_vc, L_hc]), b_c)
        c_t = f * c_tm1 + i * c_tilde
        # Output gate
        o = propup_sigmoid(T.concatenate([h_lm1_t_corrupted, h_tm1], axis=axis), T.concatenate([L_vo, L_ho]), b_o)
        # h_t
        h_t = o * T.tanh(c_t)
        return c_t, h_t

    def forward_pass(self, v, batch_size):
        input_layer = [None]*(self.n_layer+1)
        input_layer[0] = v
        n_lm1 = self.n_v

        for layer, n_h in enumerate(self.n_hs):
            c_0 = T.zeros((batch_size, n_h), dtype=theano.config.floatX)
            h_0 = T.zeros((batch_size, n_h), dtype=theano.config.floatX)
            # Infer hidden states
            (c_seq, h_seq), updates = theano.scan(fn=self.iteration,
                                                  sequences=[input_layer[layer]],
                                                  outputs_info=[c_0, h_0],
                                                  non_sequences=[self.L_vi[layer], self.L_hi[layer], self.b_i[layer],
                                                                 self.L_vf[layer], self.L_hf[layer], self.b_f[layer],
                                                                 self.L_vc[layer], self.L_hc[layer], self.b_c[layer],
                                                                 self.L_vo[layer], self.L_ho[layer], self.b_o[layer],
                                                                 n_lm1])

            # Inputs for the next layer are the hidden units of the current layer
            input_layer[layer+1] = h_seq
            # Update dimension
            n_lm1 = n_h

        # Last hidden units
        last_hidden = input_layer[self.n_layer]
        # (batch, time, pitch)
        if last_hidden.ndim == 3:
            last_hidden = last_hidden.dimshuffle((1, 0, 2))

        # Activation probability
        o_mean = propup_sigmoid(last_hidden, self.W, self.b)
        # Sample
        o_sample = self.rng.binomial(size=o_mean.shape, n=1, p=o_mean,
                                     dtype=theano.config.floatX)

        return o_mean, o_sample, updates

    ###############################
    #       COST
    ###############################
    def cost_updates(self, optimizer):
        # Time needs to be the first dimension
        v_loop = self.v.dimshuffle((1, 0, 2))

        # Infer Orchestra sequence
        self.pred, _, updates_train = self.forward_pass(v_loop, self.batch_size)

        # Compute error function
        cost = T.nnet.binary_crossentropy(self.pred, self.o)
        # Sum over time and pitch axis
        cost = cost.sum(axis=(1, 2))
        # Mean along batch dimension
        cost = T.mean(cost)

        # Weight decay
        cost = cost + self.weight_decay_coeff * self.get_weight_decay()

        # Monitor = cost normalized by sequence length
        monitor = cost / self.temporal_order

        # Update weights
        grads = T.grad(cost, self.params)
        updates_train = optimizer.get_updates(self.params, grads, updates_train)

        return cost, monitor, updates_train

    ###############################
    #       TRAIN FUNCTION
    ###############################
    def get_train_function(self, piano, orchestra, optimizer, name):
        self.step_flag = 'train'
        # index to a [mini]batch : int32
        index = T.ivector()

        # get the cost and the gradient corresponding to one step of CD-15
        cost, monitor, updates = self.cost_updates(optimizer)

        return theano.function(inputs=[index],
                               outputs=[cost, monitor],
                               updates=updates,
                               givens={self.v: build_theano_input.build_sequence(piano, index, self.batch_size, self.temporal_order, self.n_v),
                                       self.o: build_theano_input.build_sequence(orchestra, index, self.batch_size, self.temporal_order, self.n_o)},
                               name=name
                               )

    ###############################
    #       PREDICTION
    ###############################
    def prediction_measure(self):
        # Time needs to be the first dimension
        v_loop = self.v.dimshuffle((1, 0, 2))
        # Generate the last frame for the sequence v
        _, predicted_frame, updates_valid = self.forward_pass(v_loop, self.batch_size)
        # Get the ground truth
        true_frame = self.o_truth
        # Measure the performances
        precision_time = precision_measure(true_frame, predicted_frame)
        recall_time = recall_measure(true_frame, predicted_frame)
        accuracy_time = accuracy_measure(true_frame, predicted_frame)
        # 2 options :
        #       1 - take the last time index
        precision = precision_time[:, -1]
        recall = recall_time[:, -1]
        accuracy = accuracy_time[:, -1]
        #       2 - mean over time
        # precision = T.mean(precision_time, axis=1)
        # recall = T.mean(recall_time, axis=1)
        # accuracy = T.mean(accuracy_time, axis=1)
        return precision, recall, accuracy, updates_valid

    ###############################
    #       VALIDATION FUNCTION
    ##############################
    def get_validation_error(self, piano, orchestra, name):
        self.step_flag = 'validate'
        # index to a [mini]batch : int32
        index = T.ivector()

        precision, recall, accuracy, updates_valid = self.prediction_measure()

        return theano.function(inputs=[index],
                               outputs=[precision, recall, accuracy],
                               updates=updates_valid,
                               givens={self.v: build_theano_input.build_sequence(piano, index, self.batch_size, self.temporal_order, self.n_v),
                                       self.o_truth: build_theano_input.build_sequence(orchestra, index, self.batch_size, self.temporal_order, self.n_o)},
                               name=name
                               )

    ###############################
    #       GENERATION
    ###############################
    # Generation for the LSTM model is a bit special :
    # you can't seed the orchestration with the beginning of an existing score...
    def get_generate_function(self, piano, orchestra, generation_length, seed_size, batch_generation_size, name="generate_sequence"):
        self.step_flag = 'generate'

        # Index
        index = T.ivector()

        self.v_gen.tag.test_value = np.random.rand(batch_generation_size, generation_length, self.n_v).astype(theano.config.floatX)
        v_loop = self.v_gen.dimshuffle((1, 0, 2))
        _, generated_sequence, updates_generation = self.forward_pass(v_loop, batch_generation_size)

        return theano.function(inputs=[index],
                               outputs=[generated_sequence],
                               updates=updates_generation,
                               givens={self.v_gen: build_theano_input.build_sequence(piano, index, batch_generation_size, generation_length, self.n_v)},
                               name=name
                               )
