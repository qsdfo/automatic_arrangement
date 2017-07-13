#!/usr/bin/env python
# -*- coding: utf8 -*-

# Model lop
from acidano.models.lop.model_lop import Model_lop

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


class FGcRBM(Model_lop):
    """Factored Gated Conditional Restricted Boltzmann Machine (CRBM)."""

    def __init__(self,
                 model_param,
                 dimensions,
                 checksum_database,
                 weights_initialization=None):

        Model_lop.__init__(self, model_param, dimensions, checksum_database)

        self.threshold = model_param['threshold']
        self.weighted_ce = model_param['weighted_ce']

        # Datas are represented like this:
        #   - visible : (num_batch, orchestra_dim)
        #   - past : (num_batch, orchestra_dim * (temporal_order-1) + piano_dim)
        self.n_v = dimensions['orchestra_dim']
        self.n_p = (self.temporal_order-1) * dimensions['orchestra_dim']
        self.n_z = dimensions['piano_dim']

        # Number of hidden in the RBM
        self.n_h = model_param['n_hidden']
        self.n_f = model_param['n_factor']
        # Number of Gibbs sampling steps
        self.k = model_param['gibbs_steps']

        # Weights
        if weights_initialization is None:
            self.Wvf = shared_normal((self.n_v, self.n_f), 0.01, self.rng_np, name='Wvf')
            self.Whf = shared_normal((self.n_h, self.n_f), 0.01, self.rng_np, name='Whf')
            self.Wzf = shared_normal((self.n_z, self.n_f), 0.01, self.rng_np, name='Wzf')
            self.bv = shared_zeros((self.n_v), name='bv')
            self.bh = shared_zeros((self.n_h), name='bh')
            self.Apf = shared_normal((self.n_p, self.n_f), 0.01, self.rng_np, name='Apf')
            self.Avf = shared_normal((self.n_v, self.n_f), 0.01, self.rng_np, name='Avf')
            self.Azf = shared_normal((self.n_z, self.n_f), 0.01, self.rng_np, name='Azf')
            self.Bpf = shared_normal((self.n_p, self.n_f), 0.01, self.rng_np, name='Bpf')
            self.Bhf = shared_normal((self.n_h, self.n_f), 0.01, self.rng_np, name='Bhf')
            self.Bzf = shared_normal((self.n_z, self.n_f), 0.01, self.rng_np, name='Bzf')
        else:
            self.Wvf = weights_initialization['Wvf']
            self.Whf = weights_initialization['Whf']
            self.Wzf = weights_initialization['Wzf']
            self.bv = weights_initialization['bv']
            self.bh = weights_initialization['bh']
            self.Apf = weights_initialization['Apf']
            self.Avf = weights_initialization['Avf']
            self.Azf = weights_initialization['Azf']
            self.Bpf = weights_initialization['Bpf']
            self.Bhf = weights_initialization['Bhf']
            self.Bzf = weights_initialization['Bzf']

        self.params = [self.Wvf, self.Whf, self.Wzf, self.bv, self.bh, self.Apf, self.Avf, self.Azf, self.Bpf, self.Bhf, self.Bzf]

        # initialize input layer for standalone CRBM or layer0 of CDBN
        # v = orchestra(t)
        # p = past(t) = orchestra(t-N : t-1)
        # z = piano(t)
        self.v = T.matrix('v', dtype=theano.config.floatX)
        self.p = T.matrix('p', dtype=theano.config.floatX)
        self.z = T.matrix('z', dtype=theano.config.floatX)
        self.v_truth = T.matrix('v_truth', dtype=theano.config.floatX)
        self.v.tag.test_value = np.random.rand(self.batch_size, self.n_v).astype(theano.config.floatX)
        self.p.tag.test_value = np.random.rand(self.batch_size, self.n_p).astype(theano.config.floatX)
        self.z.tag.test_value = np.random.rand(self.batch_size, self.n_z).astype(theano.config.floatX)

        # Generation variables
        self.v_gen = T.matrix('v_gen', dtype=theano.config.floatX)
        self.p_gen = T.matrix('p_gen', dtype=theano.config.floatX)
        self.z_gen = T.matrix('z_gen', dtype=theano.config.floatX)

        return

    ###############################
    #       STATIC METHODS
    #       FOR METADATA AND HPARAMS
    ###############################

    @staticmethod
    def get_hp_space():

        super_space = Model_lop.get_hp_space()

        space = {'n_hidden': hopt_wrapper.qloguniform_int('n_hidden', log(3000), log(5000), 10),
                 'n_factor': hopt_wrapper.qloguniform_int('n_factor', log(3000), log(5000), 10),
                 'gibbs_steps': hopt_wrapper.qloguniform_int('gibbs_steps', log(1), log(50), 1)
                 }

        space.update(super_space)
        return space

    @staticmethod
    def name():
        return "FGcRBM"

    ###############################
    # CONDITIONAL PROBABILITIES
    ###############################
    def get_f_h(self, v, z):
        # Visible to factor : size (1,f)
        v_f = T.dot(v, self.Wvf)
        # Latent to factor : size (1,f)
        z_f = T.dot(z, self.Wzf)
        # vhl energy conditioned over hidden units
        f_h = T.dot((v_f * z_f), self.Whf.T)
        return f_h

    def get_f_v(self, h, z):
        # Hidden to factor : size (1,f)
        h_f = T.dot(h, self.Whf)
        # Latent to factor : size (1,f)
        z_f = T.dot(z, self.Wzf)
        # vhl energy conditioned over hidden units
        f_v = T.dot((h_f * z_f), self.Wvf.T)
        return f_v

    ###############################
    # DYNAMIC BIASES
    ###############################
    def get_bv_dyn(self, p, z):
        # Context to factor : size (1,f)
        p_f = T.dot(p, self.Apf)
        # Latent to factor : size (1,f)
        z_f = T.dot(z, self.Azf)
        #
        dyn_bias = self.bv + T.dot((p_f * z_f), self.Avf.T)
        return dyn_bias

    def get_bh_dyn(self, p, z):
        # Context to factor : size (1,f)
        p_f = T.dot(p, self.Bpf)
        # Latent to factor : size (1,f)
        z_f = T.dot(z, self.Bzf)
        #
        dyn_bias = self.bh + T.dot((p_f * z_f), self.Bhf.T)
        return dyn_bias

    ###############################
    # NEGATIVE PARTICLE
    ###############################
    def free_energy(self, v, z, bv, bh):
        # Get last index
        last_axis = v.ndim - 1
        # Visible contribution
        A = -(v * bv).sum(axis=last_axis)
        f_h = self.get_f_h(v, z)
        B = - T.log(1 + T.exp(f_h + bh)).sum(axis=last_axis)

        # Sum the two contributions
        fe = A + B
        return fe

    def gibbs_step(self, v, z, bv, bh, dropout_mask):
        # bv and bh defines the dynamic biases computed thanks to u_tm1
        f_h = self.get_f_h(v, z)
        mean_h = T.nnet.sigmoid(f_h + bh)
        mean_h_corrupted = T.switch(dropout_mask, mean_h, 0)
        h = self.rng.binomial(size=mean_h_corrupted.shape, n=1, p=mean_h_corrupted,
                              dtype=theano.config.floatX)
        f_v = self.get_f_v(h, z)
        mean_v = T.nnet.sigmoid(f_v + bv)
        v = self.rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                              dtype=theano.config.floatX)
        return v, mean_v

    def get_negative_particle(self, v, p, z):
        # Get dynamic biases
        bv_dyn = self.get_bv_dyn(p, z)
        bh_dyn = self.get_bh_dyn(p, z)
        # Train the RBMs by blocks
        # Dropout for RBM consists in applying the same mask to the hidden units at every gibbs sampling step
        if self.step_flag == 'train':
            dropout_mask = self.rng.binomial(size=(self.batch_size, self.n_h), n=1, p=1-self.dropout_probability, dtype=theano.config.floatX)
        elif self.step_flag in ['validate', 'generate']:
            dropout_mask = (1-self.dropout_probability)
        else:
            raise ValueError("step_flag undefined")

        # Perform k-step gibbs sampling
        (v_chain, mean_chain), updates_rbm = theano.scan(
            fn=self.gibbs_step,
            outputs_info=[v, None],
            non_sequences=[z, bv_dyn, bh_dyn, dropout_mask],
            n_steps=self.k
        )
        # Get last element of the gibbs chain
        v_sample = v_chain[-1]
        mean_v = mean_chain[-1]

        #########################
        #########################
        # Threshold ???
        if self.threshold != 0:
            idxs = (mean_v < self.threshold).nonzero()
            mean_v_clip = theano.tensor.set_subtensor(mean_v[idxs], 0)
            # Resample
            v_sample = self.rng.binomial(size=mean_v_clip.shape, n=1, p=mean_v_clip,
                                  dtype=theano.config.floatX)
        #########################
        #########################

        return v_sample, mean_v, bv_dyn, bh_dyn, updates_rbm, mean_chain

    ###############################
    #       COST
    ###############################
    def cost_updates(self, optimizer):
        # Get the negative particle
        v_sample, mean_v, bv_dyn, bh_dyn, updates_train, mean_chain = self.get_negative_particle(self.v, self.p, self.z)

        # Compute the free-energy for positive and negative particles
        fe_positive = self.free_energy(self.v, self.z, bv_dyn, bh_dyn)
        fe_negative = self.free_energy(v_sample, self.z, bv_dyn, bh_dyn)

        # Cost = mean along batches of free energy difference
        cost = T.mean(fe_positive) - T.mean(fe_negative)

        # Weight decay
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
    def build_latent(self, piano, index):
        visible = piano[index, :]
        return visible

    def build_visible(self, orchestra, index):
        visible = orchestra[index, :]
        return visible

    def build_past(self, orchestra, index):
        past_3D = build_theano_input.build_sequence(orchestra, index-1, self.batch_size, self.temporal_order-1, self.n_v)
        past = past_3D\
            .ravel()\
            .reshape((self.batch_size, (self.temporal_order-1)*self.n_v))
        return past

    def get_train_function(self, piano, orchestra, optimizer, name):
        self.step_flag = 'train'
        # index to a [mini]batch : int32
        index = T.ivector()
        index.tag.test_value = np.arange(500, 500 + self.batch_size, dtype=np.int32)

        # get the cost and the gradient corresponding to one step of CD-15
        cost, monitor, updates, mean_chain = self.cost_updates(optimizer)

        return theano.function(inputs=[index],
                               outputs=[cost, monitor],
                               updates=updates,
                               givens={self.v: self.build_visible(orchestra, index),
                                       self.p: self.build_past(orchestra, index),
                                       self.z: self.build_latent(piano, index)},
                               name=name
                               )

    ###############################
    #       PREDICTION
    ###############################
    def prediction_measure(self):
        self.v = self.rng.uniform((self.batch_size, self.n_v), 0, 1, dtype=theano.config.floatX)
        # Generate the last frame for the sequence v
        v_sample, _, _, _, updates_valid, mean_chain = self.get_negative_particle(self.v, self.p, self.z)
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
    def get_validation_error(self, piano, orchestra, name):
        self.step_flag = 'validate'
        # index to a [mini]batch : int32
        index = T.ivector()

        precision, recall, accuracy, updates_valid = self.prediction_measure()

        return theano.function(inputs=[index],
                               outputs=[precision, recall, accuracy],
                               updates=updates_valid,
                               givens={self.p: self.build_past(orchestra, index),
                                       self.z: self.build_latent(piano, index),
                                       self.v_truth: self.build_visible(orchestra, index)},
                               name=name
                               )

    ###############################
    #       GENERATION
    ###############################
    def build_p_generation(self, orchestra_gen, index, batch_size, length_seq):
        past_orchestra = orchestra_gen[:, index-self.temporal_order+1:index, :]\
            .ravel()\
            .reshape((batch_size, self.n_p))
        return past_orchestra

    def build_z_generation(self, piano_gen, index):
        present_piano = piano_gen[:, index, :]
        return present_piano

    def get_generate_function(self, piano, orchestra,
                              generation_length, seed_size, batch_generation_size,
                              name="generate_sequence"):
        self.step_flag = 'generate'

        # Seed_size is actually fixed by the temporal_order
        seed_size = self.temporal_order - 1

        # self.v_gen.tag.test_value = np.random.rand(batch_generation_size, self.n_v).astype(theano.config.floatX)
        # self.p_gen.tag.test_value = np.random.rand(batch_generation_size, self.n_p).astype(theano.config.floatX)
        # self.p_gen.tag.test_value = np.random.rand(batch_generation_size, self.n_p).astype(theano.config.floatX)

        # Graph for the negative particle
        v_sample, _, _, _, updates_next_sample, mean_chain = \
            self.get_negative_particle(self.v_gen, self.p_gen, self.z_gen)

        # Compile a function to get the next visible sample
        next_sample = theano.function(
            inputs=[self.v_gen, self.p_gen, self.z_gen],
            outputs=[v_sample],
            updates=updates_next_sample,
            name="next_sample",
        )

        def closure(ind):
            # Initialize generation matrice
            piano_gen, orchestra_gen = build_theano_input.initialization_generation(piano, orchestra, ind, generation_length, batch_generation_size, seed_size)

            for index in xrange(seed_size, generation_length, 1):
                # Build z vector = piano input
                z_gen = self.build_z_generation(piano_gen, index)
                if z_gen.sum() == 0:
                    v_t = np.zeros((self.n_v,))
                else:
                    # Build past vector
                    p_gen = self.build_p_generation(orchestra_gen, index, batch_generation_size, self.temporal_order)
                    # Build initialisation vector
                    v_gen = (np.random.uniform(0, 1, (batch_generation_size, self.n_v))).astype(theano.config.floatX)
                    # Get the next sample
                    v_t = (np.asarray(next_sample(v_gen, p_gen, z_gen))[0]).astype(theano.config.floatX)
                # Add this visible sample to the generated orchestra
                orchestra_gen[:, index, :] = v_t

            return (orchestra_gen,)

        return closure
