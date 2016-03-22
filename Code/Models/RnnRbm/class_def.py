#!/usr/bin/env python
# -*- coding: utf8 -*-

# Numpy
import numpy as np
# Theano
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
# Performance measures
from Measure.accuracy_measure import accuracy_measure
from Measure.precision_measure import precision_measure
from Measure.recall_measure import recall_measure


def shared_normal(num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(np.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))


def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(np.zeros(shape, dtype=theano.config.floatX))


class RnnRbm:
    '''Simple class to train an RNN-RBM from MIDI files and to generate sample
    sequences.'''

    def __init__(self,
                 orch=None,  # sequences as Theano matrices
                 piano=None,  # sequences as Theano matrices
                 n_orch=200,
                 n_piano=200,
                 n_hidden=500,
                 n_hidden_recurrent=490,
                 weights=(None,) * 12,
                 numpy_rng=None,
                 theano_rng=None):
        '''Constructs and compiles Theano functions for training and sequence
        generation.
        n_hidden : integer
            Number of hidden units of the conditional RBMs.
        n_hidden_recurrent : integer
            Number of hidden units of the RNN.
        lr : float
            Learning rate
        r : (integer, integer) tuple
            Specifies the pitch range of the piano-roll in MIDI note numbers,
            including r[0] but not r[1], such that r[1]-r[0] is the number of
            visible units of the RBM at a given time step. The default (21,
            109) corresponds to the full range of piano (88 notes).
        dt : float
            Sampling period when converting the MIDI files into piano-rolls, or
            equivalently the time difference between consecutive time steps.'''

        # Random generators
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        # initialize input layer for standalone CRBM or layer0 of CDBN
        self.orch = orch
        if not orch:
            self.orch = T.matrix('orch')

        self.piano = piano
        if not piano:
            self.piano = T.matrix('piano')

        self.n_orch = n_orch
        self.n_piano = n_piano
        self.n_hidden = n_hidden
        self.n_hidden_recurrent = n_hidden_recurrent

        if weights[0] is None:
            # Wuu
            self.Wuu = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.01)
            # Wuh
            self.Wuh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
            # Wuo
            self.Wuo = shared_normal(n_hidden_recurrent, n_orch, 0.0001)
            # Wup
            self.Wup = shared_normal(n_hidden_recurrent, n_piano, 0.0001)
            # Who
            self.Who = shared_normal(n_orch, n_hidden, 0.0001)
            # Whp
            self.Whp = shared_normal(n_piano, n_hidden, 0.0001)
            # Wou
            self.Wou = shared_normal(n_orch, n_hidden_recurrent, 0.0001)
            # Wpu
            self.Wpu = shared_normal(n_piano, n_hidden_recurrent, 0.0001)
            # bu
            self.bu = shared_zeros(n_hidden_recurrent)
            # bh
            self.bh = shared_zeros(n_hidden)
            # bo
            self.bo = shared_zeros(n_orch)
            # bp
            self.bp = shared_zeros(n_piano)
        else:
            self.Wuu, self.Wuh, self.Wuo, self.Wup, self.Who, self.Whp, self.Wou, self.Wpu, self.bu, self.bh, self.bo, self.bp = weights

        # learned parameters as shared variables
        self.params_RBM = self.Who, self.Whp, self.bh, self.bo, self.bp
        self.params_RNN = self.Wuu, self.Wuh, self.Wuo, self.Wup, self.Wou, self.Wpu, self.bu

    def infer_hid_seq(self):
        # Init hidden recurrent state
        u0 = T.zeros((self.n_hidden_recurrent,))

        def recurrence(o_t, p_t, u_tm1):
            bo_t = self.bo + T.dot(u_tm1, self.Wuo)
            bp_t = self.bp + T.dot(u_tm1, self.Wup)
            bh_t = self.bh + T.dot(u_tm1, self.Wuh)
            u_t = T.tanh(self.bu + T.dot(o_t, self.Wou) + T.dot(p_t, self.Wpu) + T.dot(u_tm1, self.Wuu))
            return [u_t, bo_t, bp_t, bh_t]

        # For training, the deterministic recurrence is used to compute all the
        # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
        # in batches using those parameters.
        (u_t, bo_t, bp_t, bh_t), updates = theano.scan(
            lambda o_t, p_t, u_tm1, *_: recurrence(o_t, p_t, u_tm1),
            sequences=[self.orch, self.piano],
            outputs_info=[u0, None, None, None],
            non_sequences=(self.params_RBM + self.params_RNN))

        return (u_t, bo_t, bp_t, bh_t), updates

    def CD_K(self, bo_t, bp_t, bh_t, k=1, p_clamped=None):
        # Auxiliary function for one step Gibbs sampling
        def gibbs_step(o, p):
            # hidden
            mean_h = T.nnet.sigmoid(T.dot(o, self.Who) + T.dot(p, self.Whp) + bh_t)
            h = self.theano_rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                                         dtype=theano.config.floatX)
            # orchestra and piano
            mean_o = T.nnet.sigmoid(T.dot(h, self.Who.T) + bo_t)
            orch = self.theano_rng.binomial(size=mean_o.shape, n=1, p=mean_o,
                                            dtype=theano.config.floatX)
            mean_p = T.nnet.sigmoid(T.dot(h, self.Whp.T) + bp_t)
            piano = self.theano_rng.binomial(size=mean_p.shape, n=1, p=mean_p,
                                             dtype=theano.config.floatX)

            return orch, mean_o, piano, mean_p

        if p_clamped is None:
            # Perform CD-K with the dynamic biases
            [orch_chain, mean_o_chain, piano_chain, mean_p_chain], updates = theano.scan(lambda o, p: gibbs_step(o, p),
                                                                                         outputs_info=[self.orch, None, self.piano, None],
                                                                                         n_steps=k)

            o_sample = orch_chain[-1]
            mean_o = mean_o_chain[-1]
            p_sample = piano_chain[-1]
            mean_p = mean_p_chain[-1]

        else:
            # CD-K in the case of clamped piano units
            # random initialization of the orchestra units
            orch_init = self.theano_rng.binomial(size=(self.n_orch,), n=1, p=0.5,
                                                 dtype=theano.config.floatX)
            [orch_chain_cp, mean_orch_chain_cp, _, _], updates = theano.scan(lambda o, p: gibbs_step(o, p),
                                                                             outputs_info=[orch_init, None, None, None],
                                                                             non_sequences=p_clamped,
                                                                             n_steps=k)

            orch_cp = orch_chain_cp[-1]
            mean_orch_cp = mean_orch_chain_cp[-1]

        return ([o_sample, mean_o, p_sample, mean_p], updates) \
            if (p_clamped is None) \
            else (orch_cp, mean_orch_cp, updates)

    def cost_updates(self, lr_RBM, lr_RNN, k=1):
        # Get dynamic biases
        # i.e. the influence of the chain of hidden states
        (u_t, bo_t, bp_t, bh_t), updates_train = self.infer_hid_seq()
        # Perform CD_k in the RBM
        (o_sample, mean_o, p_sample, mean_p), updates_rbm = self.CD_K(bo_t, bp_t, bh_t, k=1)

        updates_train.update(updates_rbm)

        # Monitoring
        batch_size = self.orch.shape[0]
        monitor_orch = T.xlogx.xlogy0(self.orch, mean_o) + T.xlogx.xlogy0(1 - self.orch, 1 - mean_o)
        monitor_piano = T.xlogx.xlogy0(self.piano, mean_p) + T.xlogx.xlogy0(1 - self.piano, 1 - mean_p)
        monitor = (monitor_piano.sum() + monitor_orch.sum()) / (batch_size)

        # Training cost
        def free_energy(o, p):
            return -(o * bo_t).sum() \
                - (p * bp_t).sum() \
                - T.log(1 + T.exp(T.dot(o, self.Who) + T.dot(p, self.Whp) + bh_t)).sum()
        cost = (free_energy(self.orch, self.piano) - free_energy(o_sample, p_sample)) / batch_size

        # Gradient for RBM parameters
        gparams_RBM = T.grad(cost, self.params_RBM, consider_constant=[o_sample, p_sample])
        # Updates
        for gparam, param in zip(gparams_RBM, self.params_RBM):
            # make sure that the learning rate is of the right dtype
            updates_train[param] = param - gparam * T.cast(lr_RBM, dtype=theano.config.floatX)

        # Gradient for RBM parameters
        gparams_RNN = T.grad(cost, self.params_RNN, consider_constant=[o_sample, p_sample])
        # Updates
        for gparam, param in zip(gparams_RNN, self.params_RNN):
            # make sure that the learning rate is of the right dtype
            updates_train[param] = param - gparam * T.cast(lr_RNN, dtype=theano.config.floatX)

        return cost, monitor, updates_train

    # Infer a sequence of orchestral units given a sequence of piano units
    def orchestral_inference(self, k=20):
        # Initialize the recurrent hidden states
        u0 = T.zeros((self.n_hidden_recurrent,))  # initial value for the RNN hidden units

        # TEST VALUE
        # u0.tag.test_value = np.zeros((self.n_hidden_recurrent,))  # initial value for the RNN hidden units
        # self.orch.tag.test_value = np.random.rand(self.batch_size, self.n_orch)
        # self.piano.tag.test_value = np.random.rand(self.batch_size, self.n_piano)

        def recurrence(p_t, u_tm1):
            bo_t = self.bo + T.dot(u_tm1, self.Wuo)
            bp_t = self.bp + T.dot(u_tm1, self.Wup)
            bh_t = self.bh + T.dot(u_tm1, self.Wuh)
            # o_t ? From CD-K in the RBM
            o_t, mean_o, updates_CD_K = self.CD_K(bo_t, bp_t, bh_t, k, p_clamped=p_t)
            # Infer hidden recurrent state
            u_t = T.tanh(self.bu + T.dot(o_t, self.Wou) + T.dot(p_t, self.Wpu) + T.dot(u_tm1, self.Wuu))
            return [o_t, mean_o, u_t], updates_CD_K

        (o_chain_t, mean_chain_o, u_chain_t), updates_orch_inf = theano.scan(
            recurrence,
            outputs_info=[None, None, u0],
            sequences=self.piano)

        return (o_chain_t, mean_chain_o, u_chain_t), updates_orch_inf

    def prediction_measure(self, k=20):
        """ Generate a sequence frame by frame,
            knowing the previous ground-truth at each frame prediction.
            It means the hidden recurrent units {u(t)} are computed deterministically
            with the ground-truth before the generative process
            and that they are not modified along the generative process.

            We do this to be consistent with the evaluation method used for the other models
        """
        (_, mean_o, _), updates_orch_inf = self.orchestral_inference(k)
        precision = precision_measure(self.orch, mean_o)
        recall = recall_measure(self.orch, mean_o)
        accuracy = accuracy_measure(self.orch, mean_o)
        return precision, recall, accuracy, updates_orch_inf
