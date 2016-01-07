#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Data import
import cPickle
# Numpy
import numpy as np
# Theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
# Others
import os


def train_model(model, data_path, batch_size):
    # Load datasets
    datasets = cPickle.load(data_path)
    train_set = datasets[0]
    validation_set = datasets[1]
    test_set = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set.get_value(borrow=True).shape[0] / batch_size
    n_visible = train_set.get_value(borrow=True).shape[1]

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # construct the RBM class
    rbm = RBM(input=x, n_visible=n_visible,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=15)
