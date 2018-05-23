#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:08:57 2017

@author: leo
"""

import numpy as np


def process_data_orch(orch, binarize_orch):
    ret_dict = {}
    for k, v in orch.items():
        # Binarize inputs ?
        if binarize_orch:
            v[np.nonzero(v)] = 1
        else:
            v = v / 127
        ret_dict[k] = v
    return ret_dict

def process_data_piano(piano, binarize_piano):
    ret_dict = {}
    for k, v in piano.items():
        # Binarize inputs ?
        if binarize_piano:
            v[np.nonzero(v)] = 1
        else:
            v = v / 127
        ret_dict[k] = v

    return ret_dict