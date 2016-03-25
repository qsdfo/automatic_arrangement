#!/usr/bin/env python
# -*- coding: utf8 -*-

import theano.tensor as T


# Forward propagation functions
def propup_linear(unit, W, b):
    # We use relu units
    return (T.dot(unit, W) + b)


def propup_sigmoid(unit, W, b):
    # We use relu units
    return T.nnet.sigmoid(T.dot(unit, W) + b)


def propup_relu(unit, W, b):
    # We use relu units
    return T.nnet.relu(T.dot(unit, W) + b)


def propup_tanh(unit, W, b):
    # We use relu units
    return T.tanh(T.dot(unit, W) + b)


def propup_softplus(unit, W, b):
    # We use relu units
    return T.nnet.softplus(T.dot(unit, W) + b)
