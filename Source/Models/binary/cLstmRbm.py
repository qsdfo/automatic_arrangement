#!/usr/bin/env python
# -*- coding: utf8 -*-

# Model lop
from acidano.models.lop.model_lop import Model_lop
from acidano.models.lop.binary.LSTM import LSTM

# Hyperopt
from acidano.utils import hopt_wrapper
from math import log

# Numpy
import numpy as np

# Theano
import theano
import theano.tensor as T

# Forward propagation
# Performance measures
from acidano.utils.init import shared_normal, shared_zeros
from acidano.utils.measure import accuracy_measure, precision_measure, recall_measure
# Build matrix inputs
import acidano.utils.build_theano_input as build_theano_input


class cLstmRbm(LSTM, Model_lop):
    """cLstmRbm for LOP = cRnnRbm with LSTMs.

    Predictive model,
        visible = orchestra(t)
        context = piano(t)
        cost = free-energy between positive and negative particles
    """

    def __init__(self,
                 model_param,
                 dimensions,
                 checksum_database,
                 weights_initialization=None):

        Model_lop.__init__(self, model_param, dimensions, checksum_database)

        # Number of visible units
        self.n_v = dimensions['orchestra_dim']
        # Number of context units
        self.n_c = dimensions['piano_dim']
        # Number of hidden in the RBM
        self.n_h = model_param['n_hidden']
        # Number of hidden in the recurrent net
        self.n_u = model_param['n_hidden_recurrent']
        # Number of Gibbs sampling steps
        self.k = model_param['gibbs_steps']

        # Weights
        if weights_initialization is None:
            # RBM weights
            self.W = shared_normal((self.n_v, self.n_h), 0.01, self.rng_np, name='W')
            self.bv = shared_zeros(self.n_v, name='bv')
            self.bh = shared_zeros(self.n_h, name='bh')
            # Conditional weights
            self.Wcv = shared_normal((self.n_c, self.n_v), 0.01, self.rng_np, name='Wcv')
            self.Wch = shared_normal((self.n_c, self.n_h), 0.01, self.rng_np, name='Wch')
            # Temporal biases weights
            self.Wuh = shared_normal((self.n_u, self.n_h), 0.0001, self.rng_np, name='Wuh')
            self.Wuv = shared_normal((self.n_u, self.n_v), 0.0001, self.rng_np, name='Wuv')
            # LSTM weights
            self.L_vi = shared_normal((self.n_v, self.n_u), 0.01, self.rng_np, name='L_vi')
            self.L_ui = shared_normal((self.n_u, self.n_u), 0.01, self.rng_np, name='L_ui')
            self.b_i = shared_zeros(self.n_u, name='b_i')
            self.L_vf = shared_normal((self.n_v, self.n_u), 0.01, self.rng_np, name='L_vf')
            self.L_uf = shared_normal((self.n_u, self.n_u), 0.01, self.rng_np, name='L_uf')
            self.b_f = shared_zeros(self.n_u, name='b_f')
            self.L_vstate = shared_normal((self.n_v, self.n_u), 0.01, self.rng_np, name='L_vstate')
            self.L_ustate = shared_normal((self.n_u, self.n_u), 0.01, self.rng_np, name='L_ustate')
            self.b_state = shared_zeros(self.n_u, name='b_state')
            self.L_vout = shared_normal((self.n_v, self.n_u), 0.01, self.rng_np, name='L_vout')
            self.L_uout = shared_normal((self.n_u, self.n_u), 0.01, self.rng_np, name='L_uout')
            self.b_out = shared_zeros(self.n_u, name='b_out')
        else:
            self.W = weights_initialization['W']
            self.bv = weights_initialization['bv']
            self.bh = weights_initialization['bh']
            self.Wcv = weights_initialization['Wcv']
            self.Wch = weights_initialization['Wch']
            self.Wuh = weights_initialization['Wuh']
            self.Wuv = weights_initialization['Wuv']
            self.L_vi = weights_initialization['L_vi']
            self.L_ui = weights_initialization['L_ui']
            self.b_i = weights_initialization['b_i']
            self.L_vf = weights_initialization['L_vf']
            self.L_uf = weights_initialization['L_uf']
            self.b_f = weights_initialization['b_f']
            self.L_vstate = weights_initialization['L_vstate']
            self.L_ustate = weights_initialization['L_ustate']
            self.b_state = weights_initialization['b_state']
            self.L_vout = weights_initialization['L_vout']
            self.L_uout = weights_initialization['L_uout']
            self.b_out = weights_initialization['b_out']

        self.params = [self.W, self.bv, self.bh, self.Wcv, self.Wch,
                       self.Wuh, self.Wuv,
                       self.L_vi, self.L_ui, self.b_i,
                       self.L_vf, self.L_uf, self.b_f,
                       self.L_vstate, self.L_ustate, self.b_state,
                       self.L_vout, self.L_uout, self.b_out]

        # Instanciate variables : (batch, time, pitch)
        # Note : we need the init variable to compile the theano function (get_train_function)
        # Indeed, self.v will be modified in the function, hence, giving a value to
        # self.v after these modifications does not set the value of the entrance node,
        # but set the value of the modified node
        self.v_init = T.tensor3('v', dtype=theano.config.floatX)
        self.v_init.tag.test_value = self.rng_np.rand(self.batch_size, self.temporal_order, self.n_v).astype(theano.config.floatX)
        self.c_init = T.tensor3('c', dtype=theano.config.floatX)
        self.c_init.tag.test_value = self.rng_np.rand(self.batch_size, self.temporal_order, self.n_c).astype(theano.config.floatX)
        self.v_truth = T.tensor3('v_truth', dtype=theano.config.floatX)
        self.v_truth.tag.test_value = self.rng_np.rand(self.batch_size, self.temporal_order, self.n_v).astype(theano.config.floatX)

        # Generation Variables
        self.v_seed = T.tensor3('v_seed', dtype=theano.config.floatX)
        self.c_seed = T.tensor3('c_seed', dtype=theano.config.floatX)
        self.u_gen = T.matrix('u_gen', dtype=theano.config.floatX)
        self.state_gen = T.matrix('state_gen', dtype=theano.config.floatX)
        self.c_gen = T.matrix('c_gen', dtype=theano.config.floatX)

        return

    ###############################
    #       STATIC METHODS
    #       FOR METADATA AND HPARAMS
    ###############################
    @staticmethod
    def get_hp_space():
        super_space = Model_lop.get_hp_space()

        space = {'n_hidden': hopt_wrapper.qloguniform_int('n_hidden', log(100), log(5000), 10),
                 'n_hidden_recurrent': hopt_wrapper.qloguniform_int('n_hidden_recurrent', log(100), log(5000), 10),
                 'gibbs_steps': hopt_wrapper.qloguniform_int('gibbs_steps', log(1), log(50), 1),
                 }

        space.update(super_space)
        return space

    @staticmethod
    def name():
        return "cLstmRbm"

    ###############################
    #       INFERENCE
    ###############################
    def free_energy(self, v, bv, bh):
        # sum along pitch axis (last axis)
        last_axis = v.ndim - 1
        A = -(v*bv).sum(axis=last_axis)
        C = -(T.log(1 + T.exp(T.dot(v, self.W) + bh))).sum(axis=last_axis)
        fe = A + C
        return fe

    def gibbs_step(self, v, bv, bh, dropout_mask):
        # bv and bh defines the dynamic biases computed thanks to u_tm1
        mean_h = T.nnet.sigmoid(T.dot(v, self.W) + bh)
        # Dropout
        mean_h_corrupted = T.switch(dropout_mask, mean_h, 0)
        h = self.rng.binomial(size=mean_h_corrupted.shape, n=1, p=mean_h_corrupted,
                              dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, self.W.T) + bv)
        v = self.rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                              dtype=theano.config.floatX)
        return mean_v, v

    # Given v_t, and u_tm1 we can infer u_t
    def recurrence(self, v_t, c_t, u_tm1, state_tm1):
        # Dynamic temporal biases
        bv_t = self.bv + T.dot(u_tm1, self.Wuv) + T.dot(c_t, self.Wcv)
        bh_t = self.bh + T.dot(u_tm1, self.Wuh) + T.dot(c_t, self.Wch)
        # LSTM. In our case
        # input = v_t
        # previous state = state_tm1
        # previous output = u_tm1_corrupted
        # LSTM propagation
        state_t, u_t = LSTM.iteration(self, v_t, state_tm1, u_tm1,
                                      self.L_vi, self.L_ui, self.b_i,
                                      self.L_vf, self.L_uf, self.b_f,
                                      self.L_vstate, self.L_ustate, self.b_state,
                                      self.L_vout, self.L_uout, self.b_out,
                                      self.n_v)
        return [u_t, state_t, bv_t, bh_t]

    def rnn_inference(self, v_init, c_init, u0, state_0):
        # We have to dimshuffle so that time is the first dimension
        v = v_init.dimshuffle((1, 0, 2))
        c = c_init.dimshuffle((1, 0, 2))

        # Write the recurrence to get the bias for the RBM
        (u_t, state_t, bv_t, bh_t), updates_dynamic_biases = theano.scan(
            fn=self.recurrence,
            sequences=[v, c], outputs_info=[u0, state_0, None, None])

        # Reshuffle the variables and keep trace
        self.bv_dynamic = bv_t.dimshuffle((1, 0, 2))
        self.bh_dynamic = bh_t.dimshuffle((1, 0, 2))

        # Output state_t is needed for generation step
        return u_t, state_t, updates_dynamic_biases

    def inference(self, v, c):
        # Infer the dynamic biases
        u0 = T.zeros((self.batch_size, self.n_u))  # initial value for the RNN hidden
        u0.tag.test_value = np.zeros((self.batch_size, self.n_u), dtype=theano.config.floatX)
        state_0 = T.zeros((self.batch_size, self.n_u))  # initial value for the RNN hidden
        state_0.tag.test_value = np.zeros((self.batch_size, self.n_u), dtype=theano.config.floatX)
        # Set the values for self.bv_dynamic and self.bh_dynamic
        _, _, updates_rnn_inference = self.rnn_inference(v, c, u0, state_0)

        # Train the RBMs by blocks
        # Dropout for RBM consists in applying the same mask to the hidden units at every the gibbs sampling step
        if self.step_flag == 'train':
            dropout_mask = self.rng.binomial(size=(self.batch_size, self.temporal_order, self.n_h), n=1, p=1-self.dropout_probability, dtype=theano.config.floatX)
        elif self.step_flag in ['validate', 'generate']:
            dropout_mask = (1-self.dropout_probability)
        else:
            raise ValueError("step_flag undefined")

        # Perform k-step gibbs sampling
        (mean_v_chain, v_chain), updates_inference = theano.scan(
            fn=lambda v, bv, bh: self.gibbs_step(v, bv, bh, dropout_mask),
            outputs_info=[None, v],
            non_sequences=[self.bv_dynamic, self.bh_dynamic],
            n_steps=self.k
        )

        # Add updates of the rbm
        updates_inference.update(updates_rnn_inference)

        # Get last sample of the gibbs chain
        v_sample = v_chain[-1]
        mean_v = mean_v_chain[-1]

        return v_sample, mean_v, updates_inference

    ###############################
    #       COST
    ###############################
    def cost_updates(self, optimizer):
        v_sample, mean_v, updates_train = self.inference(self.v_init, self.c_init)
        monitor_v = T.xlogx.xlogy0(self.v_init, mean_v)
        monitor = monitor_v.sum(axis=(1, 2)) / self.temporal_order
        # Mean over batches
        monitor = T.mean(monitor)

        # Compute cost function
        fe_positive = self.free_energy(self.v_init, self.bv_dynamic, self.bh_dynamic)
        fe_negative = self.free_energy(v_sample, self.bv_dynamic, self.bh_dynamic)

        # Mean along batches
        cost = T.mean(fe_positive) - T.mean(fe_negative)

        # Weight decay
        cost = cost + self.weight_decay_coeff * self.get_weight_decay()

        # Update weights
        grads = T.grad(cost, self.params, consider_constant=[v_sample])
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
                               givens={self.v_init: build_theano_input.build_sequence(orchestra, index, self.batch_size, self.temporal_order, self.n_v),
                                       self.c_init: build_theano_input.build_sequence(piano, index, self.batch_size, self.temporal_order, self.n_c)},
                               name=name
                               )

    ###############################
    #       PREDICTION
    ###############################
    def prediction_measure(self):
        self.v_init = self.rng.uniform(low=0, high=1, size=(self.batch_size, self.temporal_order, self.n_v)).astype(theano.config.floatX)
        # Generate the last frame for the sequence v
        v_sample, _, updates_valid = self.inference(self.v_init, self.c_init)
        predicted_frame = v_sample[:, -1, :]
        # Get the ground truth
        true_frame = self.v_truth[:, -1, :]
        # Measure the performances
        precision = precision_measure(true_frame, predicted_frame)
        recall = recall_measure(true_frame, predicted_frame)
        accuracy = accuracy_measure(true_frame, predicted_frame)

        return precision, recall, accuracy, updates_valid

    ###############################
    #       VALIDATION FUNCTION
    ###############################
    def get_validation_error(self, piano, orchestra, name):
        self.step_flag = 'validate'
        # index to a [mini]batch : int32
        index = T.ivector()

        precision, recall, accuracy, updates_valid = self.prediction_measure()

        return theano.function(inputs=[index],
                               outputs=[precision, recall, accuracy],
                               updates=updates_valid,
                               givens={self.v_truth: build_theano_input.build_sequence(orchestra, index, self.batch_size, self.temporal_order, self.n_v),
                                       self.c_init: build_theano_input.build_sequence(piano, index, self.batch_size, self.temporal_order, self.n_c)},
                               name=name
                               )

    ###############################
    #       GENERATION
    #   Need no seed in this model
    ###############################
    def recurrence_generation(self, c_t, u_tm1, state_tm1):
        bv_t = self.bv + T.dot(u_tm1, self.Wuv) + T.dot(c_t, self.Wcv)
        bh_t = self.bh + T.dot(u_tm1, self.Wuh) + T.dot(c_t, self.Wch)

        # Orchestra initialization
        v_init_gen = self.rng.uniform(size=(self.batch_generation_size, self.n_v), low=0.0, high=1.0).astype(theano.config.floatX)

        # Dropout for RBM consists in applying the same mask to the hidden units at every the gibbs sampling step
        if self.step_flag == 'train':
            dropout_mask = self.rng.binomial(size=(self.batch_size, self.temporal_order, self.n_h), n=1, p=1-self.dropout_probability, dtype=theano.config.floatX)
        elif self.step_flag in ['validate', 'generate']:
            dropout_mask = (1-self.dropout_probability)
        else:
            raise ValueError("step_flag undefined")

        # Inpainting :
        # p_t is clamped
        # perform k-step gibbs sampling to get o_t
        (_, v_chain), updates_inference = theano.scan(
            # Be careful argument order has been modified
            # to fit the theano function framework
            fn=lambda v, bv, bh: self.gibbs_step(v, bv, bh, dropout_mask),
            outputs_info=[None, v_init_gen],
            non_sequences=[bv_t, bh_t],
            n_steps=self.k
        )
        v_t = v_chain[-1]

        # update the lstm states
        state_t, u_t = LSTM.iteration(self, v_t, state_tm1, u_tm1,
                                      self.L_vi, self.L_ui, self.b_i,
                                      self.L_vf, self.L_uf, self.b_f,
                                      self.L_vstate, self.L_ustate, self.b_state,
                                      self.L_vout, self.L_uout, self.b_out,
                                      self.n_v)

        return u_t, state_t, v_t, updates_inference

    def get_generate_function(self, piano, orchestra,
                              generation_length, seed_size,
                              batch_generation_size,
                              name="generate_sequence"):
        self.step_flag = 'generate'

        # Seed_size is actually fixed by the temporal_order
        seed_size = self.temporal_order
        self.batch_generation_size = batch_generation_size

        ########################################################################
        #       Debug Value
        self.c_seed.tag.test_value = self.rng_np.rand(batch_generation_size, seed_size, self.n_c).astype(theano.config.floatX)
        self.v_seed.tag.test_value = self.rng_np.rand(batch_generation_size, seed_size, self.n_v).astype(theano.config.floatX)
        self.c_gen.tag.test_value = self.rng_np.rand(batch_generation_size, self.n_c).astype(theano.config.floatX)
        ########################################################################

        ########################################################################
        #       Initial hidden recurrent state (theano function)
        # Infer the state u at the end of the seed sequence
        u0 = T.zeros((batch_generation_size, self.n_u))  # initial value for the RNN hidden
        state_0 = T.zeros((batch_generation_size, self.n_u))  # initial value for the RNN states
        #########
        u0.tag.test_value = np.zeros((batch_generation_size, self.n_u), dtype=theano.config.floatX)
        state_0.tag.test_value = np.zeros((batch_generation_size, self.n_u), dtype=theano.config.floatX)
        #########
        u_chain, state_chain, updates_initialization = self.rnn_inference(self.v_seed, self.c_seed, u0, state_0)
        u_seed = u_chain[-1]
        state_seed = state_chain[-1]
        #########
        index = T.ivector()
        index.tag.test_value = [199, 1082]
        # Get the indices for the seed and generate sequences
        end_seed = index - generation_length + seed_size
        seed_function = theano.function(inputs=[index],
                                        outputs=[u_seed, state_seed],
                                        updates=updates_initialization,
                                        givens={self.v_seed: build_theano_input.build_sequence(orchestra, end_seed, batch_generation_size, seed_size, self.n_v),
                                                self.c_seed: build_theano_input.build_sequence(piano, end_seed, batch_generation_size, seed_size, self.n_c)},
                                        name=name
                                        )
        ########################################################################

        ########################################################################
        #       Next sample
        # Graph for the orchestra sample and next hidden state
        u_t, state_t, v_t, updates_next_sample = self.recurrence_generation(self.c_gen, self.u_gen, self.state_gen)
        # Compile a function to get the next visible sample
        next_sample = theano.function(
            inputs=[self.c_gen, self.u_gen, self.state_gen],
            outputs=[u_t, state_t, v_t],
            updates=updates_next_sample,
            name="next_sample",
        )
        ########################################################################

        def closure(ind):
            # Get the initial hidden chain state
            (u_t, state_t) = seed_function(ind)

            # Initialize generation matrice
            piano_gen, orchestra_gen = build_theano_input.initialization_generation(piano, orchestra, ind, generation_length, batch_generation_size, seed_size)

            for time_index in xrange(seed_size, generation_length, 1):
                # Build piano vector
                present_piano = piano_gen[:, time_index, :]
                # Next Sample and update hidden chain state
                u_t, state_t, o_t = next_sample(present_piano, u_t, state_t)
                if present_piano.sum() == 0:
                    o_t = np.zeros((self.n_orchestra,))
                # Add this visible sample to the generated orchestra
                orchestra_gen[:, time_index, :] = o_t

            return (orchestra_gen,)

        return closure
