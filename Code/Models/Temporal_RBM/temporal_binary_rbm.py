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
# Code debug and speed
import time

from Data_processing.load_data import load_data


class RBM_temporal_bin(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        input=None,
        past=None,
        n_hidden=500,
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
        self.n_visible = self.input.get_value(borrow=True).shape[0]
        self.n_past = self.past.get_value(borrow=True).shape[0]

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
        visible_chain, mean_visible_chain, past_chain, updates = theano.scan(self.gibbs_step,
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
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )

        # Monitor reconstruction (log-likelihood proxy)
        monitoring_cost = self.get_reconstruction_cost(updates, mean_neg_v)

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
        visible_chain, mean_visible_chain, past_chain, updates = theano.scan(self.gibbs_step,
                                                                             outputs_info=[self.input],
                                                                             non_sequences=[None, self.past],
                                                                             n_steps=k)

        pred_v = visible_chain[-1]
        mean_pred_v = mean_visible_chain[-1]
        return pred_v, mean_pred_v

    def prediction_measure(self, k=20):
        pred_v, mean_pred_v = self.sampling_Gibbs(k)
        precision = Score_function.prediction_measure(self.input, mean_pred_v)
        recall = Score_function.recall_measure(self.input, mean_pred_v)
        accuracy = Score_function.accuracy_measure(self.input, mean_pred_v)

        return precision, recall, accuracy


def train(hyper_parameter, dataset, log_file_path):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    """
    # Load parameters
    n_hidden = int(hyper_parameter['n_hidden'])
    temporal_order = int(hyper_parameter['temporal_order'])
    learning_rate = float(hyper_parameter['learning_rate'])
    training_epochs = int(hyper_parameter['training_epochs'])
    batch_size = int(hyper_parameter['batch_size'])

    # First check if this configuration has not been tested before,
    # i.e. its parameter are written in the result.csv file
    orch, orch_mapping, piano, piano_mapping, train_index, validate_index, test_index = load_data(data_path=dataset, log_file_path='load_log', temporal_order=temporal_order, minibatch_size=batch_size, shuffle=True, split=(0.7, 0.1, 0.2))

    # compute number of minibatches for training, validation and testing
    n_train_batches = len(train_index)
    orch_dim = orch.get_value(borrow=True).shape[1]

    #################################
    #################################
    #################################
    #####   DEBUG
    batch_index = 0
    hist_idx = np.array([train_index[batch_index] - n for n in xrange(1, temporal_order + 1)]).T
    p_test = create_past_vector(piano[train_index[batch_index]], orch[hist_idx.ravel()], batch_size, temporal_order, orch_dim)
    #################################
    #################################
    #################################

    # allocate symbolic variables for the data
    index = T.lscalar()             # index to a [mini]batch
    index_history = T.lscalar()     # index for history
    v = T.matrix('v')  # the data is presented as rasterized images
    p = T.matrix('p')  # the data is presented as rasterized images

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # construct the RBM class
    rbm = RBM_temporal_bin(input=v,
                           past=p,
                           n_hidden=n_hidden,
                           np_rng=rng,
                           theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.cost_updates(lr=learning_rate, k=1)

    free_energy = rbm.free_energy(rbm.input, rbm.past)

    #################################
    #     Training the CRBM         #
    #################################

    # the purpose of train_crbm is solely to update the CRBM parameters
    train_temp_rbm = theano.function([index, index_history],
                                     cost,
                                     updates=updates,
                                     givens={v: orch[index],
                                             p: create_past_vector(piano[index],
                                                                   orch[index_history.ravel()],
                                                                   batch_size,
                                                                   temporal_order,
                                                                   orch_dim)},
                                     name='train_temp_rbm')

    get_free_energy = theano.function([index, index_history],
                                      free_energy,
                                      givens={v: orch[index],
                                              p: create_past_vector(piano[index],
                                                                    orch[index_history.ravel()],
                                                                    batch_size,
                                                                    temporal_order,
                                                                    orch_dim)},
                                      name='get_free_energy')

    start_time = time.clock()

    # go through training epochs
    epoch = 0
    overfitting_measure = 0
    while((epoch < training_epochs) and (overfitting_measure < 0.2)):
        # go through the training set
        mean_train_cost = []
        for batch_index in xrange(n_train_batches):

            hist_idx = np.array([train_index[batch_index] - n for n in xrange(1, temporal_order + 1)]).T

            this_cost = train_temp_rbm(train_index[batch_index], hist_idx.ravel())
            # Print batch_index, this_cost
            mean_train_cost += [this_cost]

        # Validation
        all_train_idx = []
        all_val_idx = []
        for i in xrange(0, len(train_index)):
            all_train_idx.extend(train_index[i])
        all_train_hist_idx = np.array([all_train_idx - n for n in xrange(1, temporal_order + 1)]).T
        for i in xrange(0, len(validate_index)):
            all_val_idx.extend(validate_index[i])
        all_val_hist_idx = np.array([all_val_idx - n for n in xrange(1, temporal_order + 1)]).T

        free_energy_train = np.mean(get_free_energy(all_train_idx, all_train_hist_idx))
        free_energy_val = np.mean(get_free_energy(all_val_idx, all_val_hist_idx))
        overfitting_measure = (free_energy_val - free_energy_train) / free_energy_val
        print 'Training epoch %d, cost is ' % epoch, np.mean(mean_train_cost)

        epoch += 1

    end_time = time.clock()

    training_time = (end_time - start_time)

    print ('Training took %f minutes' % (training_time / 60.))

    return rbm


def create_past_vector(piano, orch, batch_size, delay, n_dim):
    # Piano is a matrix : num_batch x piano_dim
    # Orch a matrix : num_batch x ()
    import pdb; pdb.set_trace()
    orch_reshape = orch.reshape((batch_size, delay * n_dim))
    past = np.concatenate((piano, orch_reshape), axis=1)
    return past

if __name__ == '__main__':
    # Main can't be used because of relative import
    # Just here for an example of the hyperparameters structure
    # Hyper-parameter
    hyper_parameter = {}
    hyper_parameter['n_hidden'] = 500
    hyper_parameter['temporal_order'] = 10
    hyper_parameter['learning_rate'] = 0.1
    hyper_parameter['training_epochs'] = 1000
    hyper_parameter['batch_size'] = 100
    # File
    dataset = '../../../Data/data.p'
