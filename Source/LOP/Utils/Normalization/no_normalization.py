#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Normalization object
Created on Thu Nov 16 12:17:39 2017

@author: leo
"""


import numpy as np
from sklearn.decomposition import IncrementalPCA

class no_normalization(object):
    """No normalization, just let the data go through
    """
    def __init__(self, dimensions):
        self.norm_dim = dimensions['piano_input_dim']
        return

    def transform(self, matrix):
        return matrix