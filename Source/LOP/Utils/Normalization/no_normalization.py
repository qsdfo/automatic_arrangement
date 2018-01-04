#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Normalization object
Created on Thu Nov 16 12:17:39 2017

@author: leo
"""


import numpy as np
from sklearn.decomposition import IncrementalPCA
from load_matrices import load_matrices

class no_normalization(object):
    """No normalization, just let the data go through
    """
    def __init__(self, dimensions):
        self.transformed_dim = dimensions['piano_dim']
        return

    def transform(self, matrix):
        return matrix