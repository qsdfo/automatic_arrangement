#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop

# Tensorflow
import tensorflow as tf

class Random(Model_lop):
    def __init__(self, model_param, dimensions):

        Model_lop.__init__(self, model_param, dimensions)

        return

    @staticmethod
    def name():
        return "Baseline_Random"
    @staticmethod
    def binarize_piano():
        return True
    @staticmethod
    def binarize_orchestra():
        return True
    @staticmethod
    def is_keras():
        return False
    @staticmethod
    def optimize():
        return False

    @staticmethod
    def get_hp_space():
        space = Model_lop.get_hp_space()
        return space

    def predict(self, inputs_ph):
        piano_t_ph, _, _, _, _= inputs_ph
        means = tf.constant(0.5)
        dims = [tf.shape(piano_t_ph)[0], self.orch_dim]
        sample = tf.where(tf.random_uniform(dims) < means, tf.ones(dims), tf.zeros(dims))
        return sample, None