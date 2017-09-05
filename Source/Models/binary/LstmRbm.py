#!/usr/bin/env python
# -*- coding: utf8 -*-

# Model lop
from ..model_lop import Model_lop
from LSTM import LSTM

# Hyperopt
from acidano.utils import hopt_wrapper
from math import log

# Numpy
import numpy as np

# Theano
import theano
import theano.tensor as T

# Performance measures
from acidano.utils.init import shared_normal, shared_zeros
from acidano.utils.measure import accuracy_measure, precision_measure, recall_measure

# Build matrix inputs
import acidano.utils.build_theano_input as build_theano_input


class LstmRbm(LSTM, Model_lop):
    """ LstmRbm for LOP
    Predictive model,
        visible = orchestra(t) ^ piano(t)
        cost = free-energy between positive and negative particles
    """

    def __init__(self,
                 model_param,
                 dimensions,
                 checksum_database,
                 weights_initialization=None):

        Model_lop.__init__(self, model_param, dimensions, checksum_database)

        self.n_orchestra = dimensions['orchestra_dim']
        self.n_piano = dimensions['piano_dim']
        # Number of hidden in the RBM
        self.n_hidden = model_param['n_hidden']
        # Number of hidden in the recurrent net
        self.n_hidden_recurrent = model_param['n_hidden_recurrent']
        # Number of Gibbs sampling steps
        self.k = model_param['gibbs_steps']

        # Weights
        if weights_initialization is None:
            self.P = shared_normal((self.n_piano, self.n_hidden), 0.01, self.rng_np, name='P')
            self.bp = shared_zeros(self.n_piano, name='bp')
            self.O = shared_normal((self.n_orchestra, self.n_hidden), 0.01, self.rng_np, name='O')
            self.bo = shared_zeros(self.n_orchestra, name='bo')
            self.bh = shared_zeros(self.n_hidden, name='bh')
            self.Wuh = shared_normal((self.n_hidden_recurrent, self.n_hidden), 0.0001, self.rng_np, name='Wuh')
            self.Wup = shared_normal((self.n_hidden_recurrent, self.n_piano), 0.0001, self.rng_np, name='Wup')
            self.Wuo = shared_normal((self.n_hidden_recurrent, self.n_orchestra), 0.0001, self.rng_np, name='Wuo')
            self.Wpu = shared_normal((self.n_piano, self.n_hidden_recurrent), 0.0001, self.rng_np, name='Wpu')
            self.Wou = shared_normal((self.n_orchestra, self.n_hidden_recurrent), 0.0001, self.rng_np, name='Wou')
            # LSTM weights
            self.L_vi = shared_normal((self.n_orchestra+self.n_piano, self.n_hidden_recurrent), 0.01, self.rng_np, name='L_vi')
            self.L_ui = shared_normal((self.n_hidden_recurrent, self.n_hidden_recurrent), 0.01, self.rng_np, name='L_ui')
            self.b_i = shared_zeros(self.n_hidden_recurrent, name='b_i')
            self.L_vf = shared_normal((self.n_orchestra+self.n_piano, self.n_hidden_recurrent), 0.01, self.rng_np, name='L_vf')
            self.L_uf = shared_normal((self.n_hidden_recurrent, self.n_hidden_recurrent), 0.01, self.rng_np, name='L_uf')
            self.b_f = shared_zeros(self.n_hidden_recurrent, name='b_f')
            self.L_vstate = shared_normal((self.n_orchestra+self.n_piano, self.n_hidden_recurrent), 0.01, self.rng_np, name='L_vstate')
            self.L_ustate = shared_normal((self.n_hidden_recurrent, self.n_hidden_recurrent), 0.01, self.rng_np, name='L_ustate')
            self.b_state = shared_zeros(self.n_hidden_recurrent, name='b_state')
            self.L_vout = shared_normal((self.n_orchestra+self.n_piano, self.n_hidden_recurrent), 0.01, self.rng_np, name='L_vout')
            self.L_uout = shared_normal((self.n_hidden_recurrent, self.n_hidden_recurrent), 0.01, self.rng_np, name='L_uout')
            self.b_out = shared_zeros(self.n_hidden_recurrent, name='b_out')
        else:
            self.P = weights_initialization['P']
            self.bp = weights_initialization['bv']
            self.O = weights_initialization['O']
            self.bo = weights_initialization['bo']
            self.bh = weights_initialization['bh']
            self.Wuh = weights_initialization['Wuh']
            self.Wup = weights_initialization['Wup']
            self.Wuo = weights_initialization['Wuo']
            self.Wpu = weights_initialization['Wpu']
            self.Wou = weights_initialization['Wou']
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

        self.params = [self.P, self.bp, self.O, self.bo, self.bh,
                       self.Wuh, self.Wup, self.Wuo,
                       self.Wpu, self.Wou,
                       self.L_vi, self.L_ui, self.b_i,
                       self.L_vf, self.L_uf, self.b_f,
                       self.L_vstate, self.L_ustate, self.b_state,
                       self.L_vout, self.L_uout, self.b_out]


        # Instanciate variables : (batch, time, pitch)
        # Note : we need the init variable to compile the theano function (get_train_function)
        # Indeed, self.v will be modified in the function, hence, giving a value to
        # self.v after these modifications does not set the value of the entrance node,
        # but set the value of the modified node
        self.p_init = T.tensor3('v', dtype=theano.config.floatX)
        self.p_init.tag.test_value = self.rng_np.rand(self.batch_size, self.temporal_order, self.n_piano).astype(theano.config.floatX)
        self.o_init = T.tensor3('o', dtype=theano.config.floatX)
        self.o_init.tag.test_value = self.rng_np.rand(self.batch_size, self.temporal_order, self.n_orchestra).astype(theano.config.floatX)
        self.o_truth = T.tensor3('o_truth', dtype=theano.config.floatX)
        self.o_truth.tag.test_value = self.rng_np.rand(self.batch_size, self.temporal_order, self.n_orchestra).astype(theano.config.floatX)

        # Generation Variables
        self.p_seed = T.tensor3('p_seed', dtype=theano.config.floatX)
        self.o_seed = T.tensor3('o_seed', dtype=theano.config.floatX)
        self.p_gen = T.matrix('p_gen', dtype=theano.config.floatX)
        self.u_gen = T.matrix('u_gen', dtype=theano.config.floatX)
        self.state_gen = T.matrix('state_gen', dtype=theano.config.floatX)

        return

    ###############################
    ##       STATIC METHODS
    ##       FOR METADATA AND HPARAMS
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
        return "LstmRbm_inpainting"

    ###############################
    ##       INFERENCE
    ###############################
    def free_energy(self, p, o, bp, bo, bh):
        # sum along pitch axis (last axis)
        last_axis = p.ndim - 1
        A = -(p*bp).sum(axis=last_axis)
        B = -(o*bo).sum(axis=last_axis)
        C = -(T.log(1 + T.exp(T.dot(p, self.P) + T.dot(o, self.O) + bh))).sum(axis=last_axis)
        fe = A + B + C
        return fe

    def gibbs_step(self, p, o, bp, bo, bh, dropout_mask):
        # bv and bh defines the dynamic biases computed thanks to u_tm1
        mean_h = T.nnet.sigmoid(T.dot(p, self.P) + T.dot(o, self.O) + bh)
        # Dropout
        mean_h_corrupted = T.switch(dropout_mask, mean_h, 0)
        h = self.rng.binomial(size=mean_h_corrupted.shape, n=1, p=mean_h_corrupted,
                              dtype=theano.config.floatX)
        mean_p = T.nnet.sigmoid(T.dot(h, self.P.T) + bp)
        p = self.rng.binomial(size=mean_p.shape, n=1, p=mean_p,
                              dtype=theano.config.floatX)
        mean_o = T.nnet.sigmoid(T.dot(h, self.O.T) + bo)
        o = self.rng.binomial(size=mean_o.shape, n=1, p=mean_o,
                              dtype=theano.config.floatX)
        return mean_p, p, mean_o, o

    # Given v_t, and u_tm1 we can infer u_t
    def recurrence(self, p_t, o_t, u_tm1, state_tm1):
        bp_t = self.bp + T.dot(u_tm1, self.Wup)
        bo_t = self.bo + T.dot(u_tm1, self.Wuo)
        bh_t = self.bh + T.dot(u_tm1, self.Wuh)

        # LSTM propagation
        axis = p_t.ndim - 1
        v_t = T.concatenate([o_t, p_t], axis=axis)
        state_t, u_t = LSTM.iteration(self, v_t, state_tm1, u_tm1,
                                      self.L_vi, self.L_ui, self.b_i,
                                      self.L_vf, self.L_uf, self.b_f,
                                      self.L_vstate, self.L_ustate, self.b_state,
                                      self.L_vout, self.L_uout, self.b_out,
                                      (self.n_orchestra+self.n_piano))

        return [u_t, state_t, bp_t, bo_t, bh_t]

    def rnn_inference(self, p_init, o_init, u0, state_0):
        # We have to dimshuffle so that time is the first dimension
        p = p_init.dimshuffle((1,0,2))
        o = o_init.dimshuffle((1,0,2))

        # Write the recurrence to get the bias for the RBM
        (u_t, state_t, bp_t, bo_t, bh_t), updates_dynamic_biases = theano.scan(
            fn=self.recurrence,
            sequences=[p, o], outputs_info=[u0, state_0, None, None, None])

        # Reshuffle the variables
        self.bp_dynamic = bp_t.dimshuffle((1,0,2))
        self.bo_dynamic = bo_t.dimshuffle((1,0,2))
        self.bh_dynamic = bh_t.dimshuffle((1,0,2))

        return u_t, state_t, updates_dynamic_biases

    def inference(self, p, o):
        # Infer the dynamic biases
        u0 = T.zeros((self.batch_size, self.n_hidden_recurrent))  # initial value for the RNN hidden
        u0.tag.test_value = np.zeros((self.batch_size, self.n_hidden_recurrent), dtype=theano.config.floatX)
        state_0 = T.zeros((self.batch_size, self.n_hidden_recurrent))  # initial value for the RNN hidden
        state_0.tag.test_value = np.zeros((self.batch_size, self.n_hidden_recurrent), dtype=theano.config.floatX)
        _, _, updates_rnn_inference = self.rnn_inference(p, o, u0, state_0)

        # Train the RBMs by blocks
        # Dropout for RBM consists in applying the same mask to the hidden units at every the gibbs sampling step
        if self.step_flag == 'train':
            dropout_mask = self.rng.binomial(size=(self.batch_size, self.temporal_order, self.n_hidden), n=1, p=1-self.dropout_probability, dtype=theano.config.floatX)
        elif self.step_flag in ['validate', 'generate']:
            dropout_mask = (1-self.dropout_probability)
        else:
            raise ValueError("step_flag undefined")
        # Perform k-step gibbs sampling
        (mean_p_chain, p_chain, mean_o_chain, o_chain), updates_inference = theano.scan(
            fn=self.gibbs_step,
            outputs_info=[None, p, None, o],
            non_sequences=[self.bp_dynamic, self.bo_dynamic, self.bh_dynamic, dropout_mask],
            n_steps=self.k
        )

        # Add updates of the rbm
        updates_inference.update(updates_rnn_inference)

        # Get last sample of the gibbs chain
        p_sample = p_chain[-1]
        o_sample = o_chain[-1]
        mean_p = mean_p_chain[-1]
        mean_o = mean_o_chain[-1]

        return p_sample, mean_p, o_sample, mean_o, updates_inference

    ###############################
    ##       COST
    ###############################
    def cost_updates(self, optimizer):
        p_sample, mean_p, o_sample, mean_o, updates_train = self.inference(self.p_init, self.o_init)
        monitor_p = T.nnet.binary_crossentropy(self.p_init, mean_p)
        monitor_o = T.nnet.binary_crossentropy(self.o_init, mean_o)
        monitor = (T.concatenate((monitor_p, monitor_o), axis=2)).sum(axis=(1,2)) / self.temporal_order
        # Mean over batches
        monitor = T.mean(monitor)

        # Compute cost function
        fe_positive = self.free_energy(self.p_init, self.o_init, self.bp_dynamic, self.bo_dynamic, self.bh_dynamic)
        fe_negative = self.free_energy(p_sample, o_sample, self.bp_dynamic, self.bo_dynamic, self.bh_dynamic)

        # Mean along batches
        cost = T.mean(fe_positive) - T.mean(fe_negative)
        # This quantity means nothing !!
        # But by it's a gradient is close to the gradient of
        # the non-approximate cost function

        # Weight decay
        cost = cost + self.weight_decay_coeff * self.get_weight_decay()

        # Update weights
        grads = T.grad(cost, self.params, consider_constant=[p_sample, o_sample])
        updates_train = optimizer.get_updates(self.params, grads, updates_train)

        return cost, monitor, updates_train

    ###############################
    ##       TRAIN FUNCTION
    ###############################
    ###############################
    #       TRAIN FUNCTION
    ###############################
    def build_train_fn(self, optimizer, name):
        self.step_flag = 'train'
        # get the cost and the gradient corresponding to one step of CD-15
        cost, monitor, updates = self.cost_updates(optimizer)

        self.train_function = theano.function(inputs=[self.p_init, self.o_init],
                                            outputs=[cost, monitor],
                                            updates=updates,
                                            name=name
                                            )

    def train_batch(self, batch_data):
        # Simply used for parsing the batch_data
        p_init, o_init = batch_data
        return self.train_function(p_init, o_init)


    ###############################
    ##       PREDICTION
    ###############################
    def prediction_measure(self):
        self.o_init = self.rng.uniform(low=0, high=1, size=(self.batch_size, self.temporal_order, self.n_orchestra)).astype(theano.config.floatX)
        # Generate the last frame for the sequence v
        _, _, o_sample, _, updates_valid = self.inference(self.p_init, self.o_init)
        predicted_frame = o_sample[:,-1,:]
        # Get the ground truth
        true_frame = self.o_truth[:,-1,:]
        # Measure the performances
        precision = precision_measure(true_frame, predicted_frame)
        recall = recall_measure(true_frame, predicted_frame)
        accuracy = accuracy_measure(true_frame, predicted_frame)

        return precision, recall, accuracy, updates_valid

    ###############################
    ##       VALIDATION FUNCTION
    ###############################
    def build_validation_fn(self, name):
        self.step_flag = 'validate'
        
        precision, recall, accuracy, updates_valid = self.prediction_measure()

        self.validate_function = theano.function(inputs=[self.p_init, self.o_truth],
            outputs=[precision, recall, accuracy],
            updates=updates_valid,
            name=name
            )

    def validate_batch(self, batch_data):
        # Simply used for parsing the batch_data
        p_init, o_truth = batch_data
        return self.validate_function(p_init, o_truth)

    ###############################
    ##       GENERATION
    #   Need no seed in this model
    ###############################
    def recurrence_generation(self, p_t, u_tm1, state_tm1):
        bp_t = self.bp + T.dot(u_tm1, self.Wup)
        bo_t = self.bo + T.dot(u_tm1, self.Wuo)
        bh_t = self.bh + T.dot(u_tm1, self.Wuh)

        # Orchestra initialization
        o_init_gen = self.rng.uniform(size=(self.batch_generation_size, self.n_orchestra), low=0.0, high=1.0).astype(theano.config.floatX)

        # Dropout for RBM consists in applying the same mask to the hidden units at every the gibbs sampling step
        if self.step_flag == 'train':
            dropout_mask = self.rng.binomial(size=(self.batch_size, self.temporal_order, self.n_hidden), n=1, p=1-self.dropout_probability, dtype=theano.config.floatX)
        elif self.step_flag in ['generate', 'validate']:
            dropout_mask = (1-self.dropout_probability)
        else:
            raise ValueError("step_flag undefined")
        # Inpainting :
        # p_t is clamped
        # perform k-step gibbs sampling to get o_t
        (_, _, _, o_chain), updates_inference = theano.scan(
            # Be careful argument order has been modified
            # to fit the theano function framework
            fn=lambda o,p,bp,bo,bh: self.gibbs_step(p, o, bp, bo, bh, dropout_mask),
            outputs_info=[None, None, None, o_init_gen],
            non_sequences=[p_t, bp_t, bo_t, bh_t],
            n_steps=self.k
        )
        o_t = o_chain[-1]

        # LSTM update
        axis = p_t.ndim - 1
        v_t = T.concatenate([o_t, p_t], axis=axis)
        state_t, u_t = self.iteration(v_t, state_tm1, u_tm1,
                                      self.L_vi, self.L_ui, self.b_i,
                                      self.L_vf, self.L_uf, self.b_f,
                                      self.L_vstate, self.L_ustate, self.b_state,
                                      self.L_vout, self.L_uout, self.b_out,
                                      self.n_orchestra)


        return u_t, state_t, o_t, updates_inference

    def get_generate_function(self, piano, orchestra,
                              generation_length, seed_size,
                              batch_generation_size,
                              name="generate_sequence"):
        Model_lop.get_generate_function(self)
        # Seed_size is actually fixed by the temporal_order
        seed_size = self.temporal_order
        self.batch_generation_size = batch_generation_size

        ########################################################################
        #########       Test Value
        self.p_seed.tag.test_value = self.rng_np.rand(batch_generation_size, seed_size, self.n_piano).astype(theano.config.floatX)
        self.o_seed.tag.test_value = self.rng_np.rand(batch_generation_size, seed_size, self.n_orchestra).astype(theano.config.floatX)
        self.p_gen.tag.test_value = self.rng_np.rand(batch_generation_size, self.n_piano).astype(theano.config.floatX)
        self.u_gen.tag.test_value = self.rng_np.rand(batch_generation_size, self.n_hidden_recurrent).astype(theano.config.floatX)
        ########################################################################

        ########################################################################
        #########       Initial hidden recurrent state (theano function)
        # Infer the state u at the end of the seed sequence
        u0 = T.zeros((batch_generation_size, self.n_hidden_recurrent))  # initial value for the RNN hidden
        state_0 = T.zeros((batch_generation_size, self.n_hidden_recurrent))
        #########
        u0.tag.test_value = np.zeros((batch_generation_size, self.n_hidden_recurrent), dtype=theano.config.floatX)
        state_0.tag.test_value = np.zeros((batch_generation_size, self.n_hidden_recurrent), dtype=theano.config.floatX)
        #########
        u_chain, state_chain, updates_initialization = self.rnn_inference(self.p_seed, self.o_seed, u0, state_0)
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
                                        givens={self.p_seed: build_theano_input.build_sequence(piano, end_seed, batch_generation_size, seed_size, self.n_piano),
                                                self.o_seed: build_theano_input.build_sequence(orchestra, end_seed, batch_generation_size, seed_size, self.n_orchestra)},
                                        name=name
                                        )
        ########################################################################

        ########################################################################
        #########       Next sample
        # Graph for the orchestra sample and next hidden state
        u_t, state_t, o_t, updates_next_sample = self.recurrence_generation(self.p_gen, self.u_gen, self.state_gen)
        # Compile a function to get the next visible sample
        next_sample = theano.function(
            inputs=[self.p_gen, self.u_gen, self.state_gen],
            outputs=[u_t, state_t, o_t],
            updates=updates_next_sample,
            name="next_sample",
        )
        ########################################################################

        def closure(ind):
            # Get the initial hidden chain state
            (u_t, state_t) = seed_function(ind)

            # Initialize generation matrice
            piano_gen, orchestra_gen = self.initialization_generation(piano, orchestra, ind, generation_length, batch_generation_size, seed_size)

            for time_index in xrange(seed_size, generation_length, 1):
                # Build piano vector
                present_piano = piano_gen[:,time_index,:]
                # Next Sample and update hidden chain state
                u_t, state_t, o_t = next_sample(present_piano, u_t, state_t)
                if present_piano.sum() == 0:
                    o_t = np.zeros((self.n_orchestra,))
                # Add this visible sample to the generated orchestra
                orchestra_gen[:,time_index,:] = o_t

            return (orchestra_gen,)

        return closure

    def generator(self, piano, orchestra, index):
        p_init = build_theano_input.build_sequence(piano, index, self.batch_size, self.temporal_order, self.n_piano)
        o_init = build_theano_input.build_sequence(orchestra, index, self.batch_size, self.temporal_order, self.n_orchestra)
        return p_init, o_init