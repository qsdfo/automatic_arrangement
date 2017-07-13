#!/usr/bin/env python
# -*- coding: utf8 -*-

"""Repeat for binary units."""

# Model lop
from acidano.models.lop.model_lop import Model_lop

# Hyperopt
from acidano.utils import hopt_wrapper

# Theano
import theano
import theano.tensor as T

# Performance measures
from acidano.utils.measure import accuracy_measure, precision_measure, recall_measure


class Repeat(Model_lop):
    """Repeat previous frame.

    Notes : why not 100% accuracy if we just repeat o(t) instead of o(t-1) ?
    Because if the true frame is a silence and predicted frame is a silence, the score is not 1 but 0...
    """

    def __init__(self,
                 model_param,
                 dimensions,
                 checksum_database,
                 weights_initialization=None):

        Model_lop.__init__(self, model_param, dimensions, checksum_database)
        self.n_orchestra = dimensions['orchestra_dim']
        self.n_piano = dimensions['piano_dim']
        self.v_truth = T.matrix('v_truth', dtype=theano.config.floatX)
        self.past = T.matrix('past', dtype=theano.config.floatX)
        return

    ###############################
    #       STATIC METHODS
    #       FOR METADATA AND HPARAMS
    ###############################
    @staticmethod
    def get_hp_space():
        super_space = Model_lop.get_hp_space()
        # Overwrite temporal_order
        space = {'temporal_order': hopt_wrapper.quniform_int('temporal_order', 1, 1, 1)}
        space.update(super_space)
        return space

    @staticmethod
    def name():
        return "Repeat"

    ###############################
    #       TRAIN FUNCTION
    ###############################
    def get_train_function(self, piano, orchestra, optimizer, name):
        self.step_flag = 'train'

        def unit(i):
            a = 0
            b = 0
            return a, b
        return unit

    ###############################
    #       PREDICTION
    ###############################
    def prediction_measure(self):
        predicted_frame = self.past
        # Get the ground truth
        true_frame = self.v_truth
        # Measure the performances
        precision = precision_measure(true_frame, predicted_frame)
        recall = recall_measure(true_frame, predicted_frame)
        accuracy = accuracy_measure(true_frame, predicted_frame)
        updates_valid = {}
        return precision, recall, accuracy, updates_valid

    ###############################
    #       VALIDATION FUNCTION
    ##############################
    def build_visible(self, orchestra, index):
        visible = orchestra[index, :]
        return visible

    def build_past(self, orchestra, index):
        past = orchestra[index-1, :]
        return past

    def get_validation_error(self, piano, orchestra, name):
        self.step_flag = 'validate'
        # index to a [mini]batch : int32
        index = T.ivector()

        precision, recall, accuracy, updates_valid = self.prediction_measure()

        # This time self.v is initialized randomly
        return theano.function(inputs=[index],
                               outputs=[precision, recall, accuracy],
                               updates=updates_valid,
                               givens={self.v_truth: self.build_visible(orchestra, index),
                                       self.past: self.build_past(orchestra, index)},
                               name=name
                               )

    ###############################
    #       GENERATION
    ###############################
    def get_generate_function(self, piano, orchestra,
                              generation_length, seed_size, batch_generation_size,
                              name="generate_sequence"):
        self.step_flag = 'generate'

        def closure(ind):
            # Initialize generation matrice
            piano_gen, orchestra_gen = self.initialization_generation(piano, orchestra, ind, generation_length, batch_generation_size, seed_size)
            for time_index in xrange(seed_size, generation_length, 1):
                present_piano = piano_gen[:, time_index, :]
                if present_piano.sum() == 0:
                    o_t = np.zeros((self.n_orchestra,))
                else:
                    o_t = orchestra_gen[:, time_index-1, :]
                # Add this visible sample to the generated orchestra
                orchestra_gen[:, index, :] = o_t
            return (orchestra_gen,)

        return closure
