#!/usr/bin/env python
# -*- coding: utf8 -*-

""" Theano CRBM implementation.

For details, see:
http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv
Sample data:
http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/motion.mat

@author Graham Taylor"""

# Model lop
from ..model_lop import Model_lop

# Hyperopt
from acidano.utils import hopt_wrapper
from hyperopt import hp
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


class cRBM(Model_lop):
    """Conditional Restricted Boltzmann Machine (CRBM)."""

    def __init__(self,
                 model_param,
                 dimensions,
                 checksum_database,
                 weights_initialization=None):

        super(cRBM, self).__init__(model_param, dimensions, checksum_database)

        self.threshold = model_param['threshold']

        # Datas are represented like this:
        #   - visible : (num_batch, orchestra_dim)
        #   - past : (num_batch, orchestra_dim * (temporal_order-1) + piano_dim)
        self.n_piano = dimensions['piano_dim']
        self.n_orchestra = dimensions['orchestra_dim']
        #
        self.n_v = self.n_orchestra
        self.n_p = (self.temporal_order-1) * self.n_orchestra + self.n_piano
        self.n_h = model_param['n_h']

        # Number of Gibbs sampling steps
        self.k = model_param['gibbs_steps']

        # Weights
        if weights_initialization is None:
            self.W = shared_normal((self.n_v, self.n_h), 0.01, self.rng_np, name='W')
            self.bv = shared_zeros((self.n_v), name='bv')
            self.bh = shared_zeros((self.n_h), name='bh')
            self.A = shared_normal((self.n_p, self.n_v), 0.01, self.rng_np, name='A')
            self.B = shared_normal((self.n_p, self.n_h), 0.01, self.rng_np, name='B')
        else:
            self.W = weights_initialization['W']
            self.bv = weights_initialization['bv']
            self.bh = weights_initialization['bh']
            self.A = weights_initialization['A']
            self.B = weights_initialization['B']

        self.params = [self.W, self.A, self.B, self.bv, self.bh]

        # initialize input layer for standalone CRBM or layer0 of CDBN
        self.v = T.matrix('v', dtype=theano.config.floatX)
        self.p = T.matrix('p', dtype=theano.config.floatX)
        self.v_truth = T.matrix('v_truth', dtype=theano.config.floatX)

        self.v.tag.test_value = np.random.rand(self.batch_size, self.n_v).astype(theano.config.floatX)
        self.p.tag.test_value = np.random.rand(self.batch_size, self.n_p).astype(theano.config.floatX)
        
        # v_gen : random init
        # p_gen : piano[t] ^ orchestra[t-N:t-1]
        self.v_gen = T.matrix('v_gen', dtype=theano.config.floatX)
        self.p_gen = T.matrix('p_gen', dtype=theano.config.floatX)
        return

    ###############################
    #       STATIC METHODS
    #       FOR METADATA AND HPARAMS
    ###############################

    @staticmethod
    def get_hp_space():

        super_space = Model_lop.get_hp_space()

        space = {'n_h': hopt_wrapper.qloguniform_int('n_h', log(3000), log(5000), 10),
                 'gibbs_steps': hopt_wrapper.qloguniform_int('gibbs_steps', log(1), log(50), 1),
                 'threshold': hp.uniform('threshold', 0, 0.5)
                 }

        space.update(super_space)
        return space

    @staticmethod
    def name():
        return "cRBM"

    ###############################
    #       NEGATIVE PARTICLE
    ###############################
    def free_energy(self, v, bv, bh):
        # sum along pitch axis
        fe = -(v * bv).sum(axis=1) - T.log(1 + T.exp(T.dot(v, self.W) + bh)).sum(axis=1)
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
        return v, mean_v

    def get_negative_particle(self, v, p):
        # Dropout for RBM consists in applying the same mask to the hidden units at every the gibbs sampling step
        if self.step_flag == 'train':
            dropout_mask = self.rng.binomial(size=(self.batch_size, self.n_h), n=1, p=1-self.dropout_probability, dtype=theano.config.floatX)
        elif self.step_flag in ['validate', 'generate']:
            dropout_mask = (1-self.dropout_probability)
        else:
            raise ValueError("step_flag undefined")

        # Get dynamic biases
        bv_dyn = T.dot(p, self.A) + self.bv
        bh_dyn = T.dot(p, self.B) + self.bh
        # Train the RBMs by blocks
        # Perform k-step gibbs sampling
        (v_chain, mean_chain), updates_rbm = theano.scan(
            fn=self.gibbs_step,
            outputs_info=[v, None],
            non_sequences=[bv_dyn, bh_dyn, dropout_mask],
            n_steps=self.k
        )
        # Get last element of the gibbs chain
        v_sample = v_chain[-1]
        mean_v = mean_chain[-1]

        if self.threshold != 0:
            idxs = (mean_v < self.threshold).nonzero()
            mean_v_clip = theano.tensor.set_subtensor(mean_v[idxs], 0)
            # Resample
            v_sample = self.rng.binomial(size=mean_v_clip.shape, n=1, p=mean_v_clip,
                                  dtype=theano.config.floatX)

        return v_sample, mean_v, bv_dyn, bh_dyn, updates_rbm, mean_chain

    ###############################
    #       COST
    ###############################
    def cost_updates(self, optimizer):
        # Get the negative particle
        v_sample, mean_v, bv_dyn, bh_dyn, updates_train, mean_chain = self.get_negative_particle(self.v, self.p)

        # Compute the free-energy for positive and negative particles
        fe_positive = self.free_energy(self.v, bv_dyn, bh_dyn)
        fe_negative = self.free_energy(v_sample, bv_dyn, bh_dyn)

        # Cost = mean along batches of free energy difference
        cost = T.mean(fe_positive) - T.mean(fe_negative)

        # Add weight decay
        cost = cost + self.weight_decay_coeff * self.get_weight_decay()

        # Monitor
        monitor = T.nnet.binary_crossentropy(mean_v, self.v)
        monitor = monitor.sum() / self.batch_size

        # Update weights
        grads = T.grad(cost, self.params, consider_constant=[v_sample])
        updates_train = optimizer.get_updates(self.params, grads, updates_train)

        return cost, monitor, updates_train, mean_chain

    ###############################
    #       TRAIN FUNCTION
    ###############################
    def build_train_fn(self, optimizer, name):
        self.step_flag = 'train'
        # get the cost and the gradient corresponding to one step of CD-15
        cost, monitor, updates, mean_chain = self.cost_updates(optimizer)

        self.train_function = theano.function(inputs=[self.v, self.p],
            outputs=[cost, monitor],
            updates=updates,
            name=name
            )

    def train_batch(self, batch_data):
        visible, context = batch_data
        return self.train_function(visible, context)

    ###############################
    #       PREDICTION
    ###############################
    def prediction_measure(self):
        self.v = self.rng.uniform((self.batch_size, self.n_v), 0, 1, dtype=theano.config.floatX)
        # Generate the last frame for the sequence v
        v_sample, _, _, _, updates_valid, mean_chain = self.get_negative_particle(self.v, self.p)
        predicted_frame = v_sample
        # Get the ground truth
        true_frame = self.v_truth
        # Measure the performances
        precision = precision_measure(true_frame, predicted_frame)
        recall = recall_measure(true_frame, predicted_frame)
        accuracy = accuracy_measure(true_frame, predicted_frame)
        return precision, recall, accuracy, updates_valid

    ###############################
    #       VALIDATION FUNCTION
    ##############################
    def build_validation_fn(self, name):
        self.step_flag = 'validate'
        precision, recall, accuracy, updates_valid = self.prediction_measure()
        # This time self.v is initialized randomly
        self.validation_function =  theano.function(inputs=[self.v_truth, self.p],
           outputs=[precision, recall, accuracy],
           updates=updates_valid,
           name=name
           )

    def validate_batch(self, batch_data):
        # Simply used for parsing the batch_data
        visible, context = batch_data
        return self.validation_function(visible, context)

    ###############################
    #       GENERATION
    ###############################
    def build_past_generation(self, piano_gen, orchestra_gen, index, batch_size, length_seq):
        past_orchestra = orchestra_gen[:, index-self.temporal_order+1:index, :]\
            .ravel()\
            .reshape((batch_size, (length_seq-1)*self.n_v))

        present_piano = piano_gen[:, index, :]
        p_gen = np.concatenate((present_piano, past_orchestra), axis=1)
        return p_gen

    def get_generate_function(self, piano, orchestra,
                              generation_length, seed_size, batch_generation_size,
                              name="generate_sequence"):
        self.step_flag = 'generate'
        # Seed_size is actually fixed by the temporal_order
        seed_size = self.temporal_order - 1

        self.v_gen.tag.test_value = np.random.rand(batch_generation_size, self.n_v).astype(theano.config.floatX)
        self.p_gen.tag.test_value = np.random.rand(batch_generation_size, self.n_p).astype(theano.config.floatX)

        # Graph for the negative particle
        v_sample, _, _, _, updates_next_sample, mean_chain = \
            self.get_negative_particle(self.v_gen, self.p_gen)

        # Compile a function to get the next visible sample
        next_sample = theano.function(
            inputs=[self.v_gen, self.p_gen],
            outputs=[v_sample],
            updates=updates_next_sample,
            name="next_sample",
        )

        def closure(ind):
            # Initialize generation matrice
            piano_gen, orchestra_gen = build_theano_input.initialization_generation(piano, orchestra, ind, generation_length, batch_generation_size, seed_size)
            for time_index in xrange(seed_size, generation_length, 1):
                present_piano = piano_gen[:,time_index,:]
                if present_piano.sum() == 0:
                    v_t = np.zeros((self.n_orchestra,))
                else:
                    # Build past vector
                    p_gen = self.build_past_generation(piano_gen, orchestra_gen, time_index, batch_generation_size, self.temporal_order)
                    # Build initialisation vector
                    v_gen = (np.random.uniform(0, 1, (batch_generation_size, self.n_v))).astype(theano.config.floatX)
                    # Get the next sample
                    v_t = (np.asarray(next_sample(v_gen, p_gen))[0]).astype(theano.config.floatX)
                # Add this visible sample to the generated orchestra
                orchestra_gen[:, time_index, :] = v_t
            return (orchestra_gen,)
        return closure

    def generator(self, piano, orchestra, index):
        visible = orchestra[index, :]
        
        past_3D = build_theano_input.build_sequence(orchestra, index-1, self.batch_size, self.temporal_order-1, self.n_v)
        past_orchestra = past_3D\
            .ravel()\
            .reshape((self.batch_size, (self.temporal_order-1)*self.n_v))
        present_piano = piano[index, :]
        context = np.concatenate((present_piano, past_orchestra), axis=1)

        return visible, context

