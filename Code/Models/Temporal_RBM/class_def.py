#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
A temporal RBM with binary visible units.
"""
# Numpy
import numpy as np
# Theano
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from Measure.accuracy_measure import accuracy_measure
from Measure.precision_measure import precision_measure
from Measure.recall_measure import recall_measure


class RBM_temporal_bin(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        input=None,
        past=None,
        n_visible=0,
        n_hidden=500,
        n_past=0,
        W=None,
        P=None,
        hbias=None,
        vbias=None,
        pbias=None,
        np_rng=None,
        theano_rng=None
    ):

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')
        self.past = past
        if not past:
            self.past = T.matrix('past')

        # Architecture
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.n_past = n_past

        # Initialize random generators
        if np_rng is None:
            # create a number generator
            np_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = np.asarray(
                np_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + (self.n_visible + self.n_past))),
                    high=4 * np.sqrt(6. / (n_hidden + (self.n_visible + self.n_past))),
                    size=(self.n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if P is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_P = np.asarray(
                np_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + (self.n_visible + self.n_past))),
                    high=4 * np.sqrt(6. / (n_hidden + (self.n_visible + self.n_past))),
                    size=(self.n_past, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            P = theano.shared(value=initial_P, name='P', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=np.zeros(
                    self.n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        if pbias is None:
            # create shared variable for visible units bias
            pbias = theano.shared(
                value=np.zeros(
                    self.n_past,
                    dtype=theano.config.floatX
                ),
                name='pbias',
                borrow=True
            )

        self.W = W
        self.P = P
        self.hbias = hbias
        self.vbias = vbias
        self.pbias = pbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.P, self.hbias, self.vbias, self.pbias]

    def reset(self, np_rng):
        reset_W = np.asarray(
            np_rng.uniform(
                low=-4 * np.sqrt(6. / (self.n_hidden + (self.n_visible + self.n_past))),
                high=4 * np.sqrt(6. / (self.n_hidden + (self.n_visible + self.n_past))),
                size=(self.n_visible, self.n_hidden)
            ),
            dtype=theano.config.floatX
        )
        reset_P = np.asarray(
            np_rng.uniform(
                low=-4 * np.sqrt(6. / (self.n_hidden + (self.n_visible + self.n_past))),
                high=4 * np.sqrt(6. / (self.n_hidden + (self.n_visible + self.n_past))),
                size=(self.n_past, self.n_hidden)
            ),
            dtype=theano.config.floatX
        )
        self.W.set_value(reset_W)
        self.P.set_value(reset_P)
        self.hbias.set_value(np.zeros(self.n_hidden, dtype=theano.config.floatX))
        self.vbias.set_value(np.zeros(self.n_visible, dtype=theano.config.floatX))
        self.pbias.set_value(np.zeros(self.n_past, dtype=theano.config.floatX))
        return

    def free_energy(self, v, p):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v, self.W) + T.dot(p, self.P) + self.hbias
        vbias_term = T.dot(v, self.vbias) + T.dot(p, self.pbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def gibbs_step(self, v, p):
        mean_h = T.nnet.sigmoid(T.dot(v, self.W) + T.dot(p, self.P) + self.hbias)
        h = self.theano_rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                                     dtype=theano.config.floatX)
        mean_p = T.nnet.sigmoid(T.dot(h, self.P.T) + self.pbias)
        mean_v = T.nnet.sigmoid(T.dot(h, self.W.T) + self.vbias)
        v = self.theano_rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                                     dtype=theano.config.floatX)
        p = self.theano_rng.binomial(size=mean_p.shape, n=1, p=mean_p,
                                     dtype=theano.config.floatX)
        return v, mean_v, p

    # Get cost and updates for training
    def cost_updates(self, lr=0.1, k=1):
        # Negative phase
        ([visible_chain, mean_visible_chain, past_chain], updates) = theano.scan(self.gibbs_step,
                                                                                 outputs_info=[self.input, None, self.past],
                                                                                 n_steps=k)

        neg_v = visible_chain[-1]
        mean_neg_v = mean_visible_chain[-1]
        neg_p = past_chain[-1]

        # Cost
        cost = T.mean(self.free_energy(self.input, self.past)) -\
            T.mean(self.free_energy(neg_v, neg_p))

        # Gradient
        gparams = T.grad(cost, self.params, consider_constant=[neg_v, neg_p])

        # Updates
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)

        # Monitor reconstruction (log-likelihood proxy)
        monitoring_cost = self.get_reconstruction_cost(mean_neg_v)

        return monitoring_cost, updates

    def get_reconstruction_cost(self, nv):
        """Approximation to the reconstruction error """
        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(nv) +
                (1 - self.input) * T.log(1 - nv),
                axis=1
            )
        )
        return cross_entropy

    # Sampling with clamped past units
    # Two methods :
    #   - by alternate Gibbs sampling
    def sampling_Gibbs(self, k=20):
        # Negative phase with clamped past units
        ([visible_chain, mean_visible_chain, past], updates) = theano.scan(self.gibbs_step,
                                                                           outputs_info=[self.input, None, None],
                                                                           non_sequences=[self.past],
                                                                           n_steps=k)
        pred_v = visible_chain[-1]
        mean_pred_v = mean_visible_chain[-1]
        return pred_v, mean_pred_v, updates

    def prediction_measure(self, k=20):
        pred_v, mean_pred_v, updates = self.sampling_Gibbs(k)
        precision = precision_measure(self.input, mean_pred_v)
        recall = recall_measure(self.input, mean_pred_v)
        accuracy = accuracy_measure(self.input, mean_pred_v)
        return precision, recall, accuracy, updates
