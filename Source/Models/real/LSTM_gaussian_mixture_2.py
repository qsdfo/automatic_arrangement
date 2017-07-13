#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Multilayer LSTM with gaussian mixture ouput
# 1 mixture per pitch instead of 1 mixture for all the pitches.
# So instead of K weights, we now have K*O weights.
# This is supposed to make learning more difficult, but i didi this
# to prevent numerical overflows when computing likelihood
# (the normalization factor) was going to 0 very quickly with the K weights version
# This is approximately as considering each pitch independent from each others
#
# Parametric output : allows for log-likelihood computing

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
from acidano.utils.cost import gaussian_likelihood_scalar
# Sampling
from acidano.utils.sampling import gaussian_sample
# Regularization
from acidano.utils.regularization import dropout_function


class LSTM_gaussian_mixture_2(Model_lop):
    """ LSTM multi layers with gaussian mixture output for LOP
    Predictive model,
        input = piano(t)
        output = orchestra(t)
        measure = likelihood
    """

    def __init__(self,
                 model_param,
                 dimensions,
                 checksum_database,
                 weights_initialization=None):

        super(LSTM_gaussian_mixture_2, self).__init__(model_param, dimensions, checksum_database)

        self.n_v = dimensions['piano_dim']
        self.n_o = dimensions['orchestra_dim']

        # Number of hidden units
        self.n_hs = model_param['n_hidden']
        self.n_layer = len(self.n_hs)

        # Number of Gaussian in the mixture
        self.K_gaussian = model_param['K_gaussian']

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
                    n_h_lm1 = self.n_v
                else:
                    n_h_lm1 = self.n_hs[layer-1]
                n_h_l = self.n_hs[layer]
                # input gate
                self.L_vi[layer] = shared_normal((n_h_lm1, n_h_l), 0.01, name='L_vi'+str(layer))
                self.L_hi[layer] = shared_normal((n_h_l, n_h_l), 0.01, name='L_vi'+str(layer))
                self.b_i[layer] = shared_zeros((n_h_l), name='b_i'+str(layer))
                # Internal cell
                self.L_vc[layer] = shared_normal((n_h_lm1, n_h_l), 0.01, name='L_vc'+str(layer))
                self.L_hc[layer] = shared_normal((n_h_l, n_h_l), 0.01, name='L_hc'+str(layer))
                self.b_c[layer] = shared_zeros((n_h_l), name='b_c'+str(layer))
                # Forget gate
                self.L_vf[layer] = shared_normal((n_h_lm1, n_h_l), 0.01, name='L_vf'+str(layer))
                self.L_hf[layer] = shared_normal((n_h_l, n_h_l), 0.01, name='L_hf'+str(layer))
                self.b_f[layer] = shared_zeros((n_h_l), name='b_f'+str(layer))
                # Output
                # No L_co... as in Theano tuto
                self.L_vo[layer] = shared_normal((n_h_lm1, n_h_l), 0.01, name='L_vo'+str(layer))
                self.L_ho[layer] = shared_normal((n_h_l, n_h_l), 0.01, name='L_ho'+str(layer))
                self.b_o[layer] = shared_zeros((n_h_l), name='b_o'+str(layer))

            # Last layer split between three nets predicting :
            #   - mean
            #   - std
            #   - weights in the mixture
            self.W_mean = shared_normal((self.n_hs[-1], self.K_gaussian * self.n_o), 0.01, name='W_mean')
            self.b_mean = shared_zeros((self.K_gaussian * self.n_o), name='b_mean')
            self.W_std = shared_normal((self.n_hs[-1], self.K_gaussian * self.n_o), 0.01, name='W_std')
            # Special init for b_std : we don't want std to be equal to zero
            self.b_std = shared_zeros((self.K_gaussian * self.n_o), bias=0.1, name='b_std')
            self.W_weights = shared_normal((self.n_hs[-1], self.K_gaussian * self.n_o), 0.01, name='W_weights')
            self.b_weights = shared_zeros((self.K_gaussian * self.n_o), name='b_weights')
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
            self.W_mean = weights_initialization['W_mean']
            self.b_mean = weights_initialization['b_mean']
            self.W_std = weights_initialization['W_std']
            self.b_std = weights_initialization['b_std']
            self.W_weights = weights_initialization['W_weights']
            self.b_weights = weights_initialization['b_weights']

        self.params = self.L_vi.values() + self.L_hi.values() + self.b_i.values() + self.L_vc.values() + self.L_hc.values() +\
            self.b_c.values() + self.L_vf.values() + self.L_hf.values() + self.b_f.values() + self.L_vo.values() +\
            self.L_ho.values() + self.b_o.values() + [self.W_mean, self.b_mean, self.W_std, self.b_std, self.W_weights, self.b_weights]

        # Variables
        self.v = T.tensor3('v', dtype=theano.config.floatX)
        self.o = T.tensor3('o', dtype=theano.config.floatX)
        self.v_gen = T.tensor3('v_gen', dtype=theano.config.floatX)

        # Test values
        self.v.tag.test_value = np.random.rand(self.batch_size, self.temporal_order, self.n_v).astype(theano.config.floatX)
        self.o.tag.test_value = np.random.rand(self.batch_size, self.temporal_order, self.n_o).astype(theano.config.floatX)
        return

    ###############################
    ##       STATIC METHODS
    ##       FOR METADATA AND HPARAMS
    ###############################
    @staticmethod
    def get_hp_space():
        super_space = Model_lop.get_hp_space()

        space = {'n_hidden': hp.choice('n_hidden', [
            [hopt_wrapper.qloguniform_int('n_hidden_1_'+str(i), log(100), log(5000), 10) for i in range(1)],
            [hopt_wrapper.qloguniform_int('n_hidden_2_'+str(i), log(100), log(5000), 10) for i in range(2)],
            [hopt_wrapper.qloguniform_int('n_hidden_3_'+str(i), log(100), log(5000), 10) for i in range(3)],
            [hopt_wrapper.qloguniform_int('n_hidden_4_'+str(i), log(100), log(5000), 10) for i in range(4)],
        ]),
            'K_gaussian': hopt_wrapper.quniform_int('K_gaussian', 1, 10, 1),
        }

        space.update(super_space)
        return space

    @staticmethod
    def name():
        return "LSTM_ML_gaussian_mixture_2"

    ###############################
    ##  FORWARD PASS
    ###############################
    ###############################
    ##  FORWARD PASS
    ###############################
    def iteration(self, h_lm1_t, c_tm1, h_l_tm1,
                  L_vi, L_hi, b_i,
                  L_vf, L_hf, b_f,
                  L_vc, L_hc, b_c,
                  L_vo, L_ho, b_o,
                  n_lm1):
        # Sum along last axis
        axis = h_lm1_t.ndim - 1
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
            raise(TypeError("In which step are we ? Training, validation or generation ?"))
        # Input gate
        i = propup_sigmoid(T.concatenate([h_lm1_t_corrupted, h_l_tm1], axis=axis), T.concatenate([L_vi, L_hi]), b_i)
        # Forget gate
        f = propup_sigmoid(T.concatenate([h_lm1_t_corrupted, h_l_tm1], axis=axis), T.concatenate([L_vf, L_hf]), b_f)
        # Cell update term
        c_tilde = propup_tanh(T.concatenate([h_lm1_t_corrupted, h_l_tm1], axis=axis), T.concatenate([L_vc, L_hc]), b_c)
        c_t = f * c_tm1 + i * c_tilde
        # Output gate
        o = propup_sigmoid(T.concatenate([h_lm1_t_corrupted, h_l_tm1], axis=axis), T.concatenate([L_vo, L_ho]), b_o)
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
            last_hidden = last_hidden.dimshuffle((1,0,2))

        # Activation probability
        # For mean, we use a sigmoid, as the intensity is between 0 and 1 : (B,T,K*O)
        mean_mixture = T.nnet.sigmoid(T.dot(last_hidden, self.W_mean) + self.b_mean)

        # For std, we use an exponential (has to be positive) : (B,T,K*O)
        # + has the smooth property of being ~ 1 for activations ~0
        # std_mixture = T.exp(T.dot(last_hidden, self.W_std) + self.b_std)
        # NO, in fact we use ReLu
        std_mixture = T.nnet.relu(T.dot(last_hidden, self.W_std) + self.b_std)

        # Sum of the weights has to be equal to 1, so we use a softmax layer : (B,T,K*O)
        # Softmax (perso, cause theano doesn't have softmax for 3D tensor. Strange, I wonder why... ?)
        weights_activation = T.dot(last_hidden, self.W_weights) + self.b_weights
        if self.step_flag in ['train', 'validate']:
            weights_activation_reshape = T.reshape(weights_activation, (self.batch_size, self.temporal_order, self.K_gaussian, self.n_o))
        else:
            weights_activation_reshape = T.reshape(weights_activation, (self.batch_generation_size, self.generation_length, self.K_gaussian, self.n_o))
        e_x = T.exp(weights_activation_reshape - weights_activation_reshape.max(axis=2, keepdims=True))
        weights_mixture = e_x / e_x.sum(axis=2, keepdims=True)
        # weigths_prediction : (B,T,K,O)

        # Return parametric form of the gaussian mixture
        return mean_mixture, std_mixture, weights_mixture, updates

    ###############################
    ###############################
    ##       COST
    def cost(self):
        # Time needs to be the first dimension
        v_loop = self.v.dimshuffle((1,0,2))

        ###########################
        # Compute error function
        # Shuffle ground truth : (B,T,O) -> (O,B,T)
        target = self.o.dimshuffle((2,0,1))
        # Infer Orchestra sequence
        mean_prediction, std_prediction, weights_prediction, updates_train = self.forward_pass(v_loop, self.batch_size)
        # Split K and O dimension : (B,T,K*O) -> (B,T,K,O)
        mean_prediction_reshape = T.reshape(mean_prediction, (self.batch_size, self.temporal_order, self.K_gaussian, self.n_o))
        std_prediction_reshape = T.reshape(std_prediction, (self.batch_size, self.temporal_order, self.K_gaussian, self.n_o))
        # Reshuffle : (B,T,K,O) -> (K,O,B,T)
        mean_prediction_reshuffle = mean_prediction_reshape.dimshuffle((2,3,0,1))
        std_prediction_reshuffle = std_prediction_reshape.dimshuffle((2,3,0,1))

        # Note that we consider the variance matrix being a diagobnal matrix, which means that
        # their is no correlations between terms ("no rotation for the axis of the gaussians")
        # Scan over K :     (K,O,B,T) -> (K,O,B,T)
        g_ll, updates_gaussian = theano.scan(fn=lambda mean, std, target: gaussian_likelihood_scalar(target, mean, std),
                                             sequences=[mean_prediction_reshuffle, std_prediction_reshuffle],
                                             non_sequences=[target],
                                             outputs_info=[None],
                                             )

        updates_train.update(updates_gaussian)

        # Weighted sum over K gaussians -> (B,T,O)
        gaussian_mix = (weights_prediction * g_ll.dimshuffle((2,3,0,1))).sum(axis=2)

        # Take log then sum : (B,T) -> scalar
        cost = - (T.log(gaussian_mix)).sum() / (self.batch_size * self.temporal_order * self.n_o)
        ###########################

        return cost, updates_train

    def cost_updates(self, optimizer):
        cost, updates_train = self.cost()
        monitor = cost

        # Weight decay
        cost = cost + self.weight_decay_coeff * self.get_weight_decay()

        # Update weights
        grads = T.grad(cost, self.params)
        updates_train = optimizer.get_updates(self.params, grads, updates_train)
        return cost, monitor, updates_train

    ###############################
    ##       TRAIN FUNCTION
    ###############################
    def get_train_function(self, piano, orchestra, optimizer, name):

        super(LSTM_gaussian_mixture_2, self).get_train_function()

        # index to a [mini]batch : int32
        index = T.ivector()

        # get the cost and the gradient corresponding to one step of CD-15
        cost, monitor, updates = self.cost_updates(optimizer)

        return theano.function(inputs=[index],
                               outputs=[cost, monitor],
                               updates=updates,
                               givens={self.v: self.build_sequence(piano, index, self.batch_size, self.temporal_order, self.n_v),
                                       self.o: self.build_sequence(orchestra, index, self.batch_size, self.temporal_order, self.n_o)},
                               name=name
                               )

    ###############################
    ##       VALIDATION FUNCTION
    ##############################
    def get_validation_error(self, piano, orchestra, name):

        super(LSTM_gaussian_mixture_2, self).get_validation_error()

        # index to a [mini]batch : int32
        index = T.ivector()

        # For this model, and real units, use the likelihood
        # Be careful, call cost, not cost_updates since we don't want to update parameter
        cost, updates_valid = self.cost()

        return theano.function(inputs=[index],
                               outputs=[cost, cost, cost],
                               updates=updates_valid,
                               givens={self.v: self.build_sequence(piano, index, self.batch_size, self.temporal_order, self.n_v),
                                       self.o: self.build_sequence(orchestra, index, self.batch_size, self.temporal_order, self.n_o)},
                               name=name
                               )

    ###############################
    ##       GENERATION
    ###############################
    # Generation for the LSTM model is a bit special :
    # you can't seed the orchestration with the beginning of an existing score...
    def get_generate_function(self, piano, orchestra, generation_length, seed_size, batch_generation_size, name="generate_sequence"):

        super(LSTM_gaussian_mixture_2, self).get_generate_function()

        # Index
        index = T.ivector()

        # Need those values for fast-forward pass
        self.batch_generation_size = batch_generation_size
        self.generation_length = generation_length

        self.v_gen.tag.test_value = np.random.rand(batch_generation_size, generation_length, self.n_v).astype(theano.config.floatX)
        v_loop = self.v_gen.dimshuffle((1,0,2))
        mean_mixture, std_mixture, weights_generation, updates_generation = self.forward_pass(v_loop, batch_generation_size)

        # Create a mask for the 'privileged' gaussian -> (B,T,K,O)
        max_weights = T.max(weights_generation, axis=2, keepdims=True)
        mask_weights = weights_generation >= max_weights

        # Mask mean
        mean_generation_reshape = T.reshape(mean_mixture, (batch_generation_size, generation_length, self.K_gaussian, self.n_o))
        selected_mean = mean_generation_reshape * mask_weights
        mean_generation = selected_mean.sum(axis=2)

        # Mask std
        std_generation_reshape = T.reshape(std_mixture, (batch_generation_size, generation_length, self.K_gaussian, self.n_o))
        selected_std = std_generation_reshape * mask_weights
        std_generation = selected_std.sum(axis=2)

        # Sample from gaussian distribution
        sampled_gaussian = gaussian_sample(self.rng, mean_generation, std_generation)
        generated_sequence = T.clip(sampled_gaussian, 0, 1)

        return theano.function(inputs=[index],
                               outputs=[generated_sequence],
                               updates=updates_generation,
                               givens={self.v_gen: self.build_sequence(piano, index, batch_generation_size, generation_length, self.n_v)},
                               name=name
                               )
