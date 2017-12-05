#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
from LOP.Models.model_lop import Model_lop


class Zeros(Model_lop):
    def __init__(self, model_param, dimensions):
        Model_lop.__init__(self, model_param, dimensions)
        
        return

    @staticmethod
    def name():
        return "Baseline_zeros_pred"
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
         _, _, _, orch_past, _ = inputs_ph
         output = tf.zeros_like(orch_past[:,-1,:])
         return output 
