#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Models.model_lop import Model_lop


class MLFP(Model_lop):
    
    def __init__(self, model_param, dimensions):
        Model_lop.__init__(self, model_param, dimensions)
        return

    @staticmethod
    def name():
        return "Future_piano_"