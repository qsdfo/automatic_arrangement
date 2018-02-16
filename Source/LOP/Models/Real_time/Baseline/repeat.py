#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop


class Repeat(Model_lop):
    def __init__(self, model_param, dimensions):

        Model_lop.__init__(self, model_param, dimensions)

        return

    @staticmethod
    def name():
        return "Baseline_Repeat"
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
    def trainer():
        return "standard_trainer"
    @staticmethod
    def get_hp_space():
        space = Model_lop.get_hp_space()
        return space

    def predict(self, inputs_ph):
        _, _, _, orch_past_ph, _= inputs_ph
        return orch_past_ph[:,-1,:], None