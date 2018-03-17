#!/usr/bin/env python
# -*- coding: utf8 -*-

from load_matrices import load_matrices
import time


def async_load_mat(normalizer, chunk_path_list, parameters):
    """Thread for loading matrices during training
    """ 
    # Load matrix
    piano, orch, duration_piano, mask_orch = load_matrices(chunk_path_list, parameters)
    # Normalization
    piano_transformed = normalizer.transform(piano)
    return piano_transformed, orch, duration_piano, mask_orch
