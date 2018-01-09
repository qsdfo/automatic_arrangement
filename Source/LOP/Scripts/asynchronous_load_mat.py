#!/usr/bin/env python
# -*- coding: utf8 -*-

from load_matrices import load_matrices
import time


def async_load_mat(normalizer, path, parameters):
    """Thread for loading matrices during training
    """ 
    # Load matrix
    piano, orch, duration_piano, mask_orch, _ = load_matrices(path, parameters)
    # Normalization
    piano_transformed = normalizer.transform(piano)
    return piano_transformed, orch, duration_piano, mask_orch
