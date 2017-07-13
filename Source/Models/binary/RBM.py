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

# Performance measures
from acidano.utils.init import shared_normal, shared_zeros
from acidano.utils.measure import accuracy_measure, precision_measure, recall_measure


class RBM(Model_lop):
    """ A lop adapted RBM.
    Generation is performed by inpainting with some of the visible units clamped (context and piano)"""

    def __init__(self,
                 model_param,
                 dimensions,
                 checksum_database,
                 weights_initialization=None):

        super(RBM, self).__init__(model_param, dimensions, checksum_database)

        # Datas are represented like this:
        #   - visible = concatenation of the data : (num_batch, piano ^ orchestra_dim * temporal_order)
        self.n_v = dimensions['orchestra_dim']
        self.n_c = (self.temporal_order - 1) * dimensions['orchestra_dim'] + dimensions['piano_dim']

        # Number of hidden in the RBM
        self.n_h = model_param['n_hidden']
        # Number of Gibbs sampling steps
        self.k = model_param['gibbs_steps']

        # Weights
        if weights_initialization is None:
            self.W = shared_normal((self.n_v, self.n_h), 0.01, self.rng_np, name='W')
            self.C = shared_normal((self.n_c, self.n_h), 0.01, self.rng_np, name='C')
            self.bv = shared_zeros((self.n_v), name='bv')
            self.bc = shared_zeros((self.n_c), name='bc')
            self.bh = shared_zeros((self.n_h), name='bh')
        else:
            self.W = weights_initialization['W']
            self.bv = weights_initialization['bv']
            self.bc = weights_initialization['bc']
            self.C = weights_initialization['C']
            self.bh = weights_initialization['bh']

        self.params = [self.W, self.C, self.bv, self.bc, self.bh]

        # We distinguish between clamped (= C as clamped or context) and non clamped units (= V as visible units)
        self.v = T.matrix('v', dtype=theano.config.floatX)
        self.c = T.matrix('c', dtype=theano.config.floatX)
        self.v_truth = T.matrix('v_truth', dtype=theano.config.floatX)
        self.v.tag.test_value = np.random.rand(self.batch_size, self.n_v).astype(theano.config.floatX)
        self.c.tag.test_value = np.random.rand(self.batch_size, self.n_c).astype(theano.config.floatX)

        self.v_gen = T.matrix('v_gen', dtype=theano.config.floatX)
        self.c_gen = T.matrix('c_gen', dtype=theano.config.floatX)

        return

    ###############################
    ##       STATIC METHODS
    ##       FOR METADATA AND HPARAMS
    ###############################

    @staticmethod
    def get_hp_space():

        super_space = Model_lop.get_hp_space()
        # hp.qloguniform('n_hidden', log(100), log(5000), 10),
        space = {'n_hidden': hopt_wrapper.qloguniform_int('nhidden', log(3000), log(5000), 10),
                 'gibbs_steps': hopt_wrapper.qloguniform_int('gibbs_steps', log(1), log(50), 1),
                 }

        space.update(super_space)
        return space

    @staticmethod
    def name():
        return "RBM_inpainting"

    ###############################
    ##       NEGATIVE PARTICLE
    ###############################
    def free_energy(self, v, c):
        # sum along pitch axis
        A = -(v*self.bv).sum(axis=1)
        B = -(c*self.bc).sum(axis=1)
        C = -(T.log(1 + T.exp(T.dot(v, self.W) + T.dot(c, self.C) + self.bh))).sum(axis=1)
        fe = A + B + C
        return fe

    def gibbs_step(self, v, c, dropout_mask):
        # bv and bh defines the dynamic biases computed thanks to u_tm1
        mean_h = T.nnet.sigmoid(T.dot(v, self.W) + T.dot(c, self.C) + self.bh)
        mean_h_corrupted = T.switch(dropout_mask, mean_h, 0)
        h = self.rng.binomial(size=mean_h_corrupted.shape, n=1, p=mean_h_corrupted,
                              dtype=theano.config.floatX)
        v_mean = T.nnet.sigmoid(T.dot(h, self.W.T) + self.bv)
        v = self.rng.binomial(size=v_mean.shape, n=1, p=v_mean,
                              dtype=theano.config.floatX)
        c_mean = T.nnet.sigmoid(T.dot(h, self.C.T) + self.bc)
        c = self.rng.binomial(size=c_mean.shape, n=1, p=c_mean,
                              dtype=theano.config.floatX)
        return v, v_mean, c, c_mean

    def get_negative_particle(self, v, c):
        # Dropout for RBM consists in applying the same mask to the hidden units at every gibbs sampling step
        if self.step_flag == 'train':
            dropout_mask = self.rng.binomial(size=(self.batch_size, self.n_h), n=1, p=1-self.dropout_probability, dtype=theano.config.floatX)
        elif self.step_flag in ['validate', 'generate']:
            dropout_mask = (1-self.dropout_probability)
        else:
            raise ValueError("step_flag undefined")

        # Perform k-step gibbs sampling
        (v_chain, v_chain_mean, c_chain, c_chain_mean), updates_rbm = theano.scan(
            fn=lambda v,c: self.gibbs_step(v,c,dropout_mask),
            outputs_info=[v, None, c, None],
            n_steps=self.k
        )
        # Get last element of the gibbs chain
        v_sample = v_chain[-1]
        c_sample = c_chain[-1]
        _, v_mean, _, c_mean = self.gibbs_step(v_sample, c_sample, dropout_mask)

        return v_sample, v_mean, c_sample, c_mean, updates_rbm

    ###############################
    ##       COST
    ###############################
    def cost_updates(self, optimizer):
        # Get the negative particle
        v_sample, v_mean, c_sample, c_mean, updates_train = self.get_negative_particle(self.v, self.c)

        # Compute the free-energy for positive and negative particles
        fe_positive = self.free_energy(self.v, self.c)
        fe_negative = self.free_energy(v_sample, c_sample)

        # Cost = mean along batches of free energy difference
        cost = T.mean(fe_positive) - T.mean(fe_negative)

        # Weight decay
        cost = cost + self.weight_decay_coeff * self.get_weight_decay()

        # Monitor
        visible_loglike = T.nnet.binary_crossentropy(v_mean, self.v)
        context_loglike = T.nnet.binary_crossentropy(c_mean, self.c)

        # Mean over batches
        monitor = (visible_loglike.sum() + context_loglike.sum()) / self.batch_size

        # Update weights
        grads = T.grad(cost, self.params, consider_constant=[v_sample, c_sample])
        updates_train = optimizer.get_updates(self.params, grads, updates_train)

        return cost, monitor, updates_train

    ###############################
    ##       TRAIN FUNCTION
    ###############################
    def build_context(self, piano, orchestra, index):
        # [T-1, T-2, ..., 0]
        decreasing_time = theano.shared(np.arange(self.temporal_order-1,0,-1, dtype=np.int32))
        #
        temporal_shift = T.tile(decreasing_time, (self.batch_size,1))
        # Reshape
        index_full = index.reshape((self.batch_size, 1)) - temporal_shift
        # Slicing
        past_orchestra = orchestra[index_full,:]\
            .ravel()\
            .reshape((self.batch_size, (self.temporal_order-1)*self.n_v))
        present_piano = piano[index,:]
        # Concatenate along pitch dimension
        past = T.concatenate((present_piano, past_orchestra), axis=1)
        # Reshape
        return past

    def build_visible(self, orchestra, index):
        visible = orchestra[index,:]
        return visible

    def get_train_function(self, piano, orchestra, optimizer, name):
        Model_lop.get_train_function(self)
        # index to a [mini]batch : int32
        index = T.ivector()

        # get the cost and the gradient corresponding to one step of CD-15
        cost, monitor, updates = self.cost_updates(optimizer)

        return theano.function(inputs=[index],
                               outputs=[cost, monitor],
                               updates=updates,
                               givens={self.v: self.build_visible(orchestra, index),
                                       self.c: self.build_context(piano, orchestra, index)},
                               name=name
                               )

    ###############################
    ##       PREDICTION
    ###############################
    def prediction_measure(self):
        self.v = self.rng.uniform(low=0, high=1, size=(self.batch_size, self.n_v)).astype(theano.config.floatX)
        # Generate the last frame for the sequence v
        v_sample, _, c_sample, _, updates_valid = self.get_negative_particle(self.v, self.c)
        predicted_frame = v_sample
        # Get the ground truth
        true_frame = self.v_truth
        # Measure the performances
        precision = precision_measure(true_frame, predicted_frame)
        recall = recall_measure(true_frame, predicted_frame)
        accuracy = accuracy_measure(true_frame, predicted_frame)
        return precision, recall, accuracy, updates_valid

    ###############################
    ##       VALIDATION FUNCTION
    ##############################
    def get_validation_error(self, piano, orchestra, name):
        Model_lop.get_validation_error(self)
        # index to a [mini]batch : int32
        index = T.ivector()

        precision, recall, accuracy, updates_valid = self.prediction_measure()

        return theano.function(inputs=[index],
                               outputs=[precision, recall, accuracy],
                               updates=updates_valid,
                               givens={self.c: self.build_context(piano, orchestra, index),
                                       self.v_truth: self.build_visible(orchestra, index)},
                               name=name
                               )

    ###############################
    ##       GENERATION
    ###############################
    def build_c_generation(self, piano_gen, orchestra_gen, index, batch_size, length_seq):
        past_orchestra = orchestra_gen[:,index-self.temporal_order+1:index,:]\
            .ravel()\
            .reshape((batch_size, (length_seq-1)*self.n_v))
        present_piano = piano_gen[:,index,:]

        c_gen = np.concatenate((present_piano, past_orchestra), axis=1)
        return c_gen

    def get_generate_function(self, piano, orchestra,
                              generation_length, seed_size, batch_generation_size,
                              name="generate_sequence"):
        Model_lop.get_generate_function(self)
        # Seed_size is actually fixed by the temporal_order
        seed_size = self.temporal_order - 1

        # self.v_gen.tag.test_value = np.random.rand(batch_generation_size, self.n_v).astype(theano.config.floatX)
        # self.p_gen.tag.test_value = np.random.rand(batch_generation_size, self.n_c).astype(theano.config.floatX)
        # self.p_gen.tag.test_value = np.random.rand(batch_generation_size, self.n_c).astype(theano.config.floatX)

        # Graph for the negative particle
        v_sample, _, _, _, updates_next_sample = \
            self.get_negative_particle(self.v_gen, self.c_gen)

        # Compile a function to get the next visible sample
        next_sample = theano.function(
            inputs=[self.v_gen, self.c_gen],
            outputs=[v_sample],
            updates=updates_next_sample,
            name="next_sample",
        )

        def closure(ind):
            # Initialize generation matrice
            piano_gen, orchestra_gen = self.initialization_generation(piano, orchestra, ind, generation_length, batch_generation_size, seed_size)
            for time_index in xrange(seed_size, generation_length, 1):
                present_piano = piano_gen[:, time_index, :]
                if present_piano.sum() == 0:
                    # Automatically map a silence to a silence
                    v_t = np.zeros((self.n_v,))
                else:
                    # Build context vector
                    c_gen = self.build_c_generation(piano_gen, orchestra_gen, index, batch_generation_size, self.temporal_order)
                    # Build initialisation vector
                    v_gen = (np.random.uniform(0, 1, (batch_generation_size, self.n_v))).astype(theano.config.floatX)
                    # Get the next sample
                    v_t = (np.asarray(next_sample(v_gen, c_gen))[0]).astype(theano.config.floatX)
                    # Add this visible sample to the generated orchestra
                orchestra_gen[:,index,:] = v_t
            return (orchestra_gen,)

        return closure
