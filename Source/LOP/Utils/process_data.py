#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:08:57 2017

@author: leo
"""

import numpy as np


def process_data_orch(orch, parameters):
    # Binarize inputs ?
    if parameters["binarize_piano"]:
        orch[np.nonzero(orch)] = 1
    else:
        orch = orch / 127
    return orch

def process_data_piano(piano, duration_piano, parameters):
    # Binarize inputs ?
    if parameters["binarize_piano"]:
        piano[np.nonzero(piano)] = 1
    else:
        piano = piano / 127
        
    # Add duration of the piano score ?
    if parameters["duration_piano"]:
        duration_piano_reshape = np.reshape(duration_piano, [-1, 1])
        piano = np.concatenate((piano, duration_piano_reshape), axis=1)
    return piano