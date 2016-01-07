"""RBM for orchestral prediction
"""

import timeit

import numpy

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams


class RBM_orch(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        input=None,
        past=None,
        n_hidden=500,
        W=None,
        W_past=None,
        hbias=None,
        vbias=None,
        vbias_past=None,
        numpy_rng=None,
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
        self.n_visible = self.input.get_value(borrow=True).shape[0]
        self.n_past = self.past.get_value(borrow=True).shape[0]

        # Initialize random generators
        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + (self.n_visible + self.n_past))),
                    high=4 * numpy.sqrt(6. / (n_hidden + (self.n_visible + self.n_past))),
                    size=(self.n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)
        if W_past is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W_past = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + (self.n_visible + self.n_past))),
                    high=4 * numpy.sqrt(6. / (n_hidden + (self.n_visible + self.n_past))),
                    size=(self.n_past, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W_past = theano.shared(value=initial_W_past, name='W_past', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    self.n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )
        if vbias_past is None:
            # create shared variable for visible units bias
            vbias_past = theano.shared(
                value=numpy.zeros(
                    self.n_past,
                    dtype=theano.config.floatX
                ),
                name='vbias_past',
                borrow=True
            )

        self.W = W
        self.W_past = W_past
        self.hbias = hbias
        self.vbias = vbias
        self.vbias_past = vbias_past
        self.theano_rng = theano_rng
        self.params = [self.W, self.W_past, self.hbias, self.vbias, self.vbias_past]

        def free_energy(self, v_sample, v_sample_past):
            ''' Function to compute the free energy '''
            v_sample
            wx_b = T.dot(v_sample, self.W) + T.dot(v_sample_past, self.W_past) + self.hbias
            vbias_term = T.dot(v_sample, self.vbias) + T.dot(v_sample_past, self.vbias_past)
            hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
            return -hidden_term - vbias_term

        def propup(self, vis, vis_past):
            pre_sigmoid_activation = T.dot(vis, self.W) + T.dot(vis_past, self.W_past) + self.hbias
            return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

        def sample_h_given_v(self, v0_sample, v0_sample_past):
            # compute the activation of the hidden units given a sample of
            # the visibles
            pre_sigmoid_h1, h1_mean = self.propup(v0_sample, v0_sample_past)
            # get a sample of the hiddens given their activation
            # Note that theano_rng.binomial returns a symbolic sample of dtype
            # int64 by default. If we want to keep our computations in floatX
            # for the GPU we need to specify to return the dtype floatX
            h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                                 n=1, p=h1_mean,
                                                 dtype=theano.config.floatX)
            return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        pre_sigmoid_activation_past = T.dot(hid, self.W_past.T) + self.vbias_past
        return [pre_sigmoid_activation, pre_sigmoid_activation_past, T.nnet.sigmoid(pre_sigmoid_activation), T.nnet.sigmoid(pre_sigmoid_activation_past)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]
