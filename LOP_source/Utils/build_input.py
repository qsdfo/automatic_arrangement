#!/usr/bin/env python
# -*- coding: utf8 -*-

import theano
import theano.tensor as T
import numpy as np

def build_sequence(pr, index, batch_size, seq_length, last_dim):
    # T = seq_length
    # [T-1, T-2, ..., 0]
    decreasing_time = np.arange(seq_length-1, -1, -1, dtype=np.int32)
    # Temporal_shift =
    #
    #        [i0-T+1   ; i1-T+1; i2-T+1 ; ... ; iN-T+1;
    #         i0-T+2 ;                  ; iN-T+2;
    #                       ...
    #         i0 ;                      ; iN]
    #
    #   with T = temporal_order
    #        N = pitch_order
    #
    temporal_shift = np.tile(decreasing_time, (batch_size, 1))
    # Reshape
    index_full = index.reshape((batch_size, 1)) - temporal_shift
    # Slicing
    pr = pr[index_full.ravel(), :]
    # Reshape
    return np.reshape(pr, (batch_size, seq_length, last_dim))


######
# Those functions are used for generating sequences
# with originally non-sequential models
# such as RBM, cRBM, FGcRBM...
######
def build_seed(pr, index, batch_size, length_seq):
    n_dim = len(pr.shape)
    last_dim = pr.shape[n_dim-1]
    # [T-1, T-2, ..., 0]
    decreasing_time = np.arange(length_seq-1, -1, -1, dtype=np.int32)
    #
    temporal_shift = np.tile(decreasing_time, (batch_size, 1))
    # Reshape
    index_broadcast = np.expand_dims(index, axis=1)
    index_full = index_broadcast - temporal_shift
    # Slicing
    seed_pr = pr[index_full, :]\
        .ravel()\
        .reshape((batch_size, length_seq, last_dim))
    return seed_pr


def initialization_generation(piano, orchestra, ind, generation_length, batch_generation_size, seed_size):
    # Build piano generation
    piano_gen = build_seed(piano, ind,
                           batch_generation_size, generation_length)

    # Build orchestra seed and cast it in the orchestration generation vector
    first_generated_ind = (ind - generation_length + seed_size) + 1
    last_orchestra_seed_ind = first_generated_ind - 1
    orchestra_seed = build_seed(orchestra, last_orchestra_seed_ind,
                                batch_generation_size, seed_size)

    n_orchestra = orchestra.shape[1]
    orchestra_gen = np.zeros((batch_generation_size, generation_length, n_orchestra)).astype(theano.config.floatX)
    orchestra_gen[:, :seed_size, :] = orchestra_seed
    return piano_gen, orchestra_gen
