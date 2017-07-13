#!/usr/bin/env python
# -*- coding: utf8 -*-

""" Categorical cRBM """

# Model lop
from acidano.models.lop.categorical.categorical_lop_model import Categorical_lop_model
from acidano.models.lop.binary.cRBM import cRBM

# Numpy
import numpy as np

# Theano
import theano
import theano.tensor as T

from acidano.utils.init import shared_zeros
# Performance measures
from acidano.utils.measure import accuracy_measure_categorical, precision_measure_categorical, recall_measure_categorical


class cRBM_cat(Categorical_lop_model, cRBM):
    """Categorical Conditional Restricted Boltzmann Machine (CRBM) """
    def __init__(self,
                 model_param,
                 dimensions,
                 checksum_database,
                 weights_initialization=None):

        Categorical_lop_model.__init__(self, model_param, dimensions, checksum_database)
        cRBM.__init__(self, model_param, dimensions, checksum_database)
        return

    ###############################
    ##       STATIC METHODS
    ##       FOR METADATA AND HPARAMS
    ###############################

    # herited methods :
    #   get_hp_space()
    #   name():

    ###############################
    ##       NEGATIVE PARTICLE
    ###############################
    # Free energy is not even modified !! :)
    #     free_energy(self, v, bv, bh):

    def gibbs_step(self, v, bv, bh, dropout_mask):
        import pdb; pdb.set_trace()
        # bv and bh defines the dynamic biases computed thanks to u_tm1
        mean_h = T.nnet.sigmoid(T.dot(v, self.W) + bh)
        # Dropout
        mean_h_corrupted = T.switch(dropout_mask, mean_h, 0)
        h = self.rng.binomial(size=mean_h_corrupted.shape, n=1, p=mean_h_corrupted,
                              dtype=theano.config.floatX)

        mean_v = T.nnet.sigmoid(T.dot(h, self.W.T) + bv)
        # Softmax normalization
        # Sum per blocks of size self.N_cat
        # Divide
        mean_v_reshape = T.reshape(mean_v, (self.batch_size, self.n_v/self.N_cat, self.N_cat))
        norm = T.sum(mean_v_reshape, axis=2, keepdims=True)
        mean_v_softmax_badshape = mean_v_reshape / norm
        mean_v_softmax = T.reshape(mean_v_softmax_badshape, (self.batch_size, self.n_v))

        # Special sampling
        max_v_softmax = T.max(mean_v_softmax, axis=2, keepdims=True)
        v = mean_v_softmax == max_v_softmax
        return v, mean_v

    #     get_negative_particle(self, v, p)

    ###############################
    ##       COST
    ###############################
    #    cost_updates(self, optimizer):

    ###############################
    ##       TRAIN FUNCTION
    ###############################
    #     get_index_full(self, index, batch_size, length_seq)
    #     build_past(self, piano, orchestra, index, batch_size, length_seq)
    #     build_visible(self, orchestra, index)
    #     get_train_function(self, piano, orchestra, optimizer, name)

    ###############################
    ##       PREDICTION
    ###############################
    def prediction_measure(self):
        ############ Visible initialization during generation
        ##### We just set all units to the same value so that none of them is favored
        self.v = shared_zeros((self.batch_size, self.n_v), bias=1/self.N_cat, name=None)
        # Generate the last frame for the sequence v
        v_sample, _, _, _, updates_valid = self.get_negative_particle(self.v, self.p)
        predicted_frame = v_sample
        # Get the ground truth
        true_frame = self.v_truth
        # Measure the performances
        precision = precision_measure_categorical(true_frame, predicted_frame)
        recall = recall_measure_categorical(true_frame, predicted_frame)
        accuracy = accuracy_measure_categorical(true_frame, predicted_frame)
        return precision, recall, accuracy, updates_valid

    ###############################
    ##       VALIDATION FUNCTION
    ##############################
    #     get_validation_error(self, piano, orchestra, name)

    ###############################
    ##       GENERATION
    ###############################
    #    build_past_generation(self, piano_gen, orchestra_gen, index, batch_size, length_seq):

    @cRBM.generate_flag
    def get_generate_function(self, piano, orchestra,
                              generation_length, seed_size, batch_generation_size,
                              name="generate_sequence"):
        # Seed_size is actually fixed by the temporal_order
        seed_size = self.temporal_order - 1

        self.v_gen.tag.test_value = np.random.rand(batch_generation_size, self.n_v).astype(theano.config.floatX)
        self.p_gen.tag.test_value = np.random.rand(batch_generation_size, self.n_p).astype(theano.config.floatX)

        # Graph for the negative particle
        v_sample, _, _, _, updates_next_sample = \
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
            piano_gen, orchestra_gen = self.initialization_generation(piano, orchestra, ind, generation_length, batch_generation_size, seed_size)
            for index in xrange(seed_size, generation_length, 1):
                # Build past vector
                p_gen = self.build_past_generation(piano_gen, orchestra_gen, index, batch_generation_size, self.temporal_order)
                # Build initialisation vector
                ######## Respect categorical structure when initializing
                ##### We just set all units to the same value so that none of them is favored
                v_gen = shared_zeros((self.batch_size, self.n_v), bias=1/self.N_cat, name=None).astype(theano.config.floatX)
                # Get the next sample
                v_t = (np.asarray(next_sample(v_gen, p_gen))[0]).astype(theano.config.floatX)
                # Add this visible sample to the generated orchestra
                orchestra_gen[:,index,:] = v_t
            return (orchestra_gen,)

        return closure
