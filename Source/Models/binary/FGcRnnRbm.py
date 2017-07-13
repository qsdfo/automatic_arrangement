#!/usr/bin/env python
# -*- coding: utf8 -*-

# Model lop
from acidano.models.lop.binary.FGcRBM import FGcRBM
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


class FGcRnnRbm(FGcRBM, Model_lop):
    """Factored Gated Conditional Restricted Boltzmann Machine (CRBM)."""

    def __init__(self,
                 model_param,
                 dimensions,
                 checksum_database,
                 weights_initialization=None):

        Model_lop.__init__(self, model_param, dimensions, checksum_database)

        # Datas are represented like this:
        #   - visible : (num_batch, orchestra_dim)
        #   - past : (num_batch, orchestra_dim * (temporal_order-1) + piano_dim)
        self.n_orchestra = dimensions['orchestra_dim']
        self.n_piano = dimensions['piano_dim']

        # Number of hidden in the RBM
        self.n_factor = model_param['n_factor']
        self.n_hidden = model_param['n_hidden']
        # Number of hidden in the recurrent net
        self.n_hidden_recurrent = model_param['n_hidden_recurrent']
        # Number of Gibbs sampling steps
        self.k = model_param['gibbs_steps']

        # Weights
        if weights_initialization is None:
            self.Wvf = shared_normal((self.n_orchestra, self.n_factor), 0.001, self.rng_np, name='Wvf')
            self.Whf = shared_normal((self.n_hidden, self.n_factor), 0.001, self.rng_np, name='Whf')
            self.Wzf = shared_normal((self.n_piano, self.n_factor), 0.001, self.rng_np, name='Wzf')
            self.bv = shared_zeros((self.n_orchestra), name='bv')
            self.bh = shared_zeros((self.n_hidden), name='bh')
            self.Apf = shared_normal((self.n_hidden_recurrent, self.n_factor), 0.001, self.rng_np, name='Apf')
            self.Avf = shared_normal((self.n_orchestra, self.n_factor), 0.001, self.rng_np, name='Avf')
            self.Azf = shared_normal((self.n_piano, self.n_factor), 0.001, self.rng_np, name='Azf')
            self.Bpf = shared_normal((self.n_hidden_recurrent, self.n_factor), 0.001, self.rng_np, name='Bpf')
            self.Bhf = shared_normal((self.n_hidden, self.n_factor), 0.001, self.rng_np, name='Bhf')
            self.Bzf = shared_normal((self.n_piano, self.n_factor), 0.001, self.rng_np, name='Bzf')
            self.Wpp = shared_normal((self.n_hidden_recurrent, self.n_hidden_recurrent), 0.0001, self.rng_np, name='Wpp')
            self.Wvp = shared_normal((self.n_orchestra, self.n_hidden_recurrent), 0.0001, self.rng_np, name='Wvp')
            self.bp = shared_zeros(self.n_hidden_recurrent, name='bp')
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
            self.Wpp = weights_initialization['Wpp']
            self.Wvp = weights_initialization['Wvp']
            self.bp = weights_initialization['bp']

        self.params = [self.Wvf, self.Whf, self.Wzf, self.bv, self.bh, self.Apf, self.Avf, self.Azf, self.Bpf, self.Bhf, self.Bzf, self.Wpp, self.Wvp, self.bp]

        # Instanciate variables : (batch, time, pitch)
        # Note : we need the init variable to compile the theano function (get_train_function)
        # Indeed, self.v will be modified in the function, hence, giving a value to
        # self.v after these modifications does not set the value of the entrance node,
        # but set the value of the modified node
        self.v_init = T.tensor3('v', dtype=theano.config.floatX)
        self.v_init.tag.test_value = self.rng_np.rand(self.batch_size, self.temporal_order, self.n_orchestra).astype(theano.config.floatX)
        self.z_init = T.tensor3('z', dtype=theano.config.floatX)
        self.z_init.tag.test_value = self.rng_np.rand(self.batch_size, self.temporal_order, self.n_piano).astype(theano.config.floatX)
        self.v_truth = T.tensor3('v_truth', dtype=theano.config.floatX)
        self.v_truth.tag.test_value = self.rng_np.rand(self.batch_size, self.temporal_order, self.n_orchestra).astype(theano.config.floatX)

        # Generation Variables
        self.v_seed = T.tensor3('v_seed', dtype=theano.config.floatX)
        self.z_seed = T.tensor3('z_seed', dtype=theano.config.floatX)
        self.z_gen = T.matrix('z_gen', dtype=theano.config.floatX)
        self.p_gen = T.matrix('p_gen', dtype=theano.config.floatX)

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
                 'n_factor': hopt_wrapper.qloguniform_int('n_factor', log(100), log(5000), 10),
                 'gibbs_steps': hopt_wrapper.qloguniform_int('gibbs_steps', log(1), log(50), 1),
                 }

        space.update(super_space)
        return space

    @staticmethod
    def name():
        return "FGcRnnRbm"

    ###############################
    # CONDITIONAL PROBABILITIES and DYNAMIC BIASES
    ###############################
    # FGcRBM.get_f_h(v, z)
    # FGcRBM.get_f_v(h, z)
    # FGcRBM.get_bv_dyn(p, z)
    # FGcRBM.get_bh_dyn(p, z)

    ###############################
    #       FGcRBM INFERENCE
    ###############################
    # bv and bh are dynamic biases in the folowing definitions
    # FGcRBM.free_energy(v, z, bv, bh)
    # FGcRBM.gibbs_step(v, z, bv, bh, dropout_mask)

    ###############################
    #       RNN CHAIN
    ###############################
    # Given v_t, and p_tm1 we can infer p_t
    def recurrence(self, v_t, z_t, p_tm1):
        bv_t = FGcRBM.get_bv_dyn(self, p_tm1, z_t)
        bh_t = FGcRBM.get_bh_dyn(self, p_tm1, z_t)
        p_t = T.tanh(self.bp + T.dot(v_t, self.Wvp) + T.dot(p_tm1, self.Wpp))
        return [p_t, bv_t, bh_t]

    def rnn_inference(self, v_init, z_init, p0):
        # We have to dimshuffle so that time is the first dimension
        v = v_init.dimshuffle((1, 0, 2))
        z = z_init.dimshuffle((1, 0, 2))

        # Write the recurrence to get the bias for the RBM
        (p_t, bv_t, bh_t), updates_dynamic_biases = theano.scan(
            fn=self.recurrence,
            sequences=[v, z],
            outputs_info=[p0, None, None]
        )

        # Reshuffle the variables
        bv_dyn = bv_t.dimshuffle((1, 0, 2))
        bh_dyn = bh_t.dimshuffle((1, 0, 2))

        return p_t, bv_dyn, bh_dyn, updates_dynamic_biases

    ###############################
    #       NEGATIVE PARTICLE IN THE RBM
    ###############################
    def get_negative_particle(self, v, z):
        # Get dynamic biases
        p0 = T.zeros((self.batch_size, self.n_hidden_recurrent))  # initial value for the RNN hidden
        p0.tag.test_value = np.zeros((self.batch_size, self.n_hidden_recurrent), dtype=theano.config.floatX)
        _, bv_dyn, bh_dyn, updates_rnn_inference = self.rnn_inference(v, z, p0)

        # Train the FGcRBMs by blocks
        # Dropout for RBM consists in applying the same mask to the hidden units at every the gibbs sampling step
        if self.step_flag == 'train':
            dropout_mask = self.rng.binomial(size=(self.batch_size, self.temporal_order, self.n_hidden), n=1, p=1-self.dropout_probability, dtype=theano.config.floatX)
        elif self.step_flag in ['validate', 'generate']:
            dropout_mask = (1-self.dropout_probability)
        else:
            raise ValueError("step_flag undefined")

        # Perform k-step gibbs sampling
        (v_chain, mean_chain), updates_inference = theano.scan(
            fn=self.gibbs_step,
            outputs_info=[v, None],
            non_sequences=[z, bv_dyn, bh_dyn, dropout_mask],
            n_steps=self.k
        )

        # Add updates of the rbm
        updates_inference.update(updates_rnn_inference)

        # Get last sample of the gibbs chain
        v_sample = v_chain[-1]
        mean_v = mean_chain[-1]

        return v_sample, mean_v, bv_dyn, bh_dyn, updates_inference

    ###############################
    #       COST
    ###############################
    def cost_updates(self, optimizer):
        v_sample, mean_v, bv_dyn, bh_dyn, updates_train = self.get_negative_particle(self.v_init, self.z_init)
        monitor = (T.nnet.binary_crossentropy(mean_v, self.v_init)).sum(axis=(1, 2))/self.temporal_order
        # Mean over batches
        monitor = T.mean(monitor)

        # Compute cost function
        fe_positive = FGcRBM.free_energy(self, self.v_init, self.z_init, bv_dyn, bh_dyn)
        fe_negative = FGcRBM.free_energy(self, v_sample, self.z_init, bv_dyn, bh_dyn)

        # Mean along batches
        cost = T.mean(fe_positive) - T.mean(fe_negative)
        # This quantity means nothing !!
        # But by its gradient is close to the gradient of
        # the true cost function

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
                               givens={self.v_init: build_theano_input.build_sequence(orchestra, index, self.batch_size, self.temporal_order, self.n_orchestra),
                                       self.z_init: build_theano_input.build_sequence(piano, index, self.batch_size, self.temporal_order, self.n_piano)},
                               name=name
                               )

    ###############################
    #       PREDICTION
    ###############################
    def prediction_measure(self):
        self.v_init = self.rng.uniform(low=0, high=1, size=(self.batch_size, self.temporal_order, self.n_orchestra)).astype(theano.config.floatX)
        # Generate the last frame for the sequence v
        v_sample, _, _, _, updates_valid = self.get_negative_particle(self.v_init, self.z_init)
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
        self.step_flag = 'validation'

        # index to a [mini]batch : int32
        index = T.ivector()

        precision, recall, accuracy, updates_valid = self.prediction_measure()

        return theano.function(inputs=[index],
                               outputs=[precision, recall, accuracy],
                               updates=updates_valid,
                               givens={self.z_init: build_theano_input.build_sequence(piano, index, self.batch_size, self.temporal_order, self.n_piano),
                                       self.v_truth: build_theano_input.build_sequence(orchestra, index, self.batch_size, self.temporal_order, self.n_orchestra)},
                               name=name
                               )

    ###############################
    #       GENERATION
    #   Need no seed in this model
    ###############################
    def recurrence_generation(self, z_t, p_tm1):
        bv_dyn = FGcRBM.get_bv_dyn(self, p_tm1, z_t)
        bh_dyn = FGcRBM.get_bh_dyn(self, p_tm1, z_t)

        # Orchestra initialization
        v_init_gen = self.rng.uniform(size=(self.batch_generation_size, self.n_orchestra), low=0.0, high=1.0).astype(theano.config.floatX)

        # Dropout for RBM consists in applying the same mask to the hidden units at every the gibbs sampling step
        if self.step_flag == 'train':
            dropout_mask = self.rng.binomial(size=(self.batch_size, self.temporal_order, self.n_hidden), n=1, p=1-self.dropout_probability, dtype=theano.config.floatX)
        elif self.step_flag in ['validate', 'generate']:
            dropout_mask = (1-self.dropout_probability)
        else:
            raise ValueError("step_flag undefined")

        # Get the negative particle
        # Perform k-step gibbs sampling
        (v_chain, _), updates_inference = theano.scan(
            fn=self.gibbs_step,
            outputs_info=[v_init_gen, None],
            non_sequences=[z_t, bv_dyn, bh_dyn, dropout_mask],
            n_steps=self.k
        )
        v_t = v_chain[-1]

        # update the rnn state
        p_t = T.tanh(self.bp + T.dot(v_t, self.Wvp) + T.dot(p_tm1, self.Wpp))

        return p_t, v_t, updates_inference

    def get_generate_function(self, piano, orchestra,
                              generation_length, seed_size,
                              batch_generation_size,
                              name="generate_sequence"):
        self.step_flag = 'generate'

        # Seed_size is actually fixed by the temporal_order
        seed_size = self.temporal_order
        self.batch_generation_size = batch_generation_size

        ########################################################################
        #     Test Value
        self.z_seed.tag.test_value = self.rng_np.rand(batch_generation_size, seed_size, self.n_piano).astype(theano.config.floatX)
        self.v_seed.tag.test_value = self.rng_np.rand(batch_generation_size, seed_size, self.n_orchestra).astype(theano.config.floatX)
        self.z_gen.tag.test_value = self.rng_np.rand(batch_generation_size, self.n_piano).astype(theano.config.floatX)
        self.p_gen.tag.test_value = self.rng_np.rand(batch_generation_size, self.n_hidden_recurrent).astype(theano.config.floatX)
        ########################################################################

        ########################################################################
        #      Initial hidden recurrent state (theano function)
        # Infer the state u at the end of the seed sequence
        p0 = T.zeros((batch_generation_size, self.n_hidden_recurrent))  # initial value for the RNN hidden
        #########
        p0.tag.test_value = np.zeros((batch_generation_size, self.n_hidden_recurrent), dtype=theano.config.floatX)
        #########
        p_chain, _, _, updates_initialization = self.rnn_inference(self.v_seed, self.z_seed, p0)
        p_seed = p_chain[-1]
        index = T.ivector()
        index.tag.test_value = [199, 1082]
        # Get the indices for the seed and generate sequences
        end_seed = index - generation_length + seed_size
        seed_function = theano.function(inputs=[index],
                                        outputs=[p_seed],
                                        updates=updates_initialization,
                                        givens={self.z_seed: build_theano_input.build_sequence(piano, end_seed, batch_generation_size, seed_size, self.n_piano),
                                                self.v_seed: build_theano_input.build_sequence(orchestra, end_seed, batch_generation_size, seed_size, self.n_orchestra)},
                                        name=name
                                        )
        ########################################################################

        ########################################################################
        #      Next sample
        # Graph for the orchestra sample and next hidden state
        next_p, next_v, updates_next_sample = self.recurrence_generation(self.z_gen, self.p_gen)
        # Compile a function to get the next visible sample
        next_sample = theano.function(
            inputs=[self.z_gen, self.p_gen],
            outputs=[next_p, next_v],
            updates=updates_next_sample,
            name="next_sample",
        )
        ########################################################################

        def closure(ind):
            # Get the initial hidden chain state
            (p_t,) = seed_function(ind)

            # Initialize generation matrice
            piano_gen, orchestra_gen = build_theano_input.initialization_generation(piano, orchestra, ind, generation_length, batch_generation_size, seed_size)

            for time_index in xrange(seed_size, generation_length, 1):
                # Build piano vector
                present_piano = piano_gen[:, time_index, :]
                # Next Sample and update hidden chain state
                p_t, v_t = next_sample(present_piano, p_t)
                if present_piano.sum() == 0:
                    v_t = np.zeros((self.n_orchestra,))
                # Add this visible sample to the generated orchestra
                orchestra_gen[:, time_index, :] = v_t

            return (orchestra_gen,)

        return closure
