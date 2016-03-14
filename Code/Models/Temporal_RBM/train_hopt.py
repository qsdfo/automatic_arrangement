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
# Hyperopt
from hyperopt import hp, fmin, tpe
from math import log
# CSV
import csv
# Code debug and speed
import time

from Data_processing.load_data import load_data_tvt
from Models.Temporal_RBM.class_def import RBM_temporal_bin


def train_hopt(temporal_granularity, dataset, max_evals, log_file_path, csv_file_path):
    # Init log_file
    with open(log_file_path, 'wb') as log_file:
        log_file.write((u'# TRBM : Hyperoptimization : \n').encode('utf8'))
        log_file.write((u'# Temporal granularity : ' + temporal_granularity + '\n').encode('utf8'))

    # Define hyper-parameter search space
    header = ['n_hidden', 'temporal_order', 'learning_rate', 'batch_size']
    space = (hp.qloguniform('n_hidden', log(100), log(5000), 10),
             hp.qloguniform('temporal_order', log(1), log(30), 1),
             hp.uniform('learning_rate', 0.0001, 1),
             hp.quniform('batch_size', 10, 500, 10)
             #  hp.choice('activation_func', ['tanh', 'sigmoid']),
             #  hp.choice('sampling_positive', ['true', 'false'])
             # gibbs_sampling_step_test ???
             )

    # Hyperparameter parameters.... haha, we should optimize them (joke).
    global run_counter
    run_counter = 0

    def run_wrapper(params):
        global run_counter
        run_counter += 1
        # log
        with open(log_file_path, 'wb') as log_file:
            log_file.write((u'\n###################').encode('utf8'))
            log_file.write((u'# Config :  {}'.format(run_counter)).encode('utf8'))
        # print
        print((u'\n###################').encode('utf8'))
        print((u'# Config :  {}'.format(run_counter)).encode('utf8'))

        # Train ##############
        error = train(params, dataset, temporal_granularity, log_file_path)
        ######################

        # log
        with open(log_file_path, 'wb') as log_file:
            log_file.write((u'# Error :  {}\n'.format(error)).encode('utf8'))
            log_file.write((u'###################\n').encode('utf8'))
        # print
        print((u'# Error :  {}\n'.format(error)).encode('utf8'))
        print((u'###################\n').encode('utf8'))

        # Write the result in result.csv
        with open(csv_file_path, 'ab') as csvfile:
            n_hidden, temporal_order, learning_rate, batch_size = params
            writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=header)
            dico_res = {'n_hidden': n_hidden,
                        'temporal_order': temporal_order,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size}
            writer.writerow(dico_res)

        return error

    with open(csv_file_path, 'wb') as csvfile:
        # Write headers if they don't already exist
        writerHead = csv.writer(csvfile, delimiter=',')
        writerHead.writerow(header)

    start_time = time.clock()
    best = fmin(run_wrapper, space, algo=tpe.suggest, max_evals=max_evals)
    end_time = time.clock()

    print (start_time - end_time)

    return best


def train(params, dataset, temporal_granularity, log_file_path):
    # Hyperparams
    # n_hidden, temporal_order, learning_rate, \
    #     batch_size, activation_func, sampling_positive = params
    n_hidden, temporal_order, learning_rate, batch_size = params

    # Cast the hp
    n_hidden = int(n_hidden)
    temporal_order = int(temporal_order)
    batch_size = int(batch_size)

    # Log them
    with open(log_file_path, 'wb') as log_file:
        log_file.write((u'# n_hidden :  {}'.format(n_hidden)).encode('utf8'))
        log_file.write((u'# temporal_order :  {}'.format(temporal_order)).encode('utf8'))
        log_file.write((u'# learning_rate :  {}'.format(learning_rate)).encode('utf8'))
        log_file.write((u'# batch_size :  {}'.format(batch_size)).encode('utf8'))
    # Print
    print((u'# n_hidden :  {}'.format(n_hidden)).encode('utf8'))
    print((u'# temporal_order :  {}'.format(temporal_order)).encode('utf8'))
    print((u'# learning_rate :  {}'.format(learning_rate)).encode('utf8'))
    print((u'# batch_size :  {}'.format(batch_size)).encode('utf8'))

    # Some hp not optimized
    gibbs_sampling_step_test = 40

    # Load data
    orch, orch_mapping, piano, piano_mapping, train_index, val_index, _ \
        = load_data_tvt(data_path=dataset,
                        log_file_path='bullshit.txt',
                        temporal_granularity=temporal_granularity,
                        temporal_order=temporal_order,
                        shared_bool=True,
                        minibatch_size=batch_size,
                        split=(0.7, 0.1, 0.2))

    n_train_batches = train_index.shape[0]
    last_batch_size = (train_index[-1]).shape[0]

    n_val_batches = len(val_index)
    val_size = 0
    for i in xrange(n_val_batches):
        val_size += val_index[i].size

    # Get dimensions
    orch_dim = orch.get_value(borrow=True).shape[1]
    n_past = (piano.get_value(borrow=True).shape[1]) + (orch.get_value(borrow=True).shape[1]) * temporal_order

    # allocate symbolic variables for the data
    index = T.lvector()             # index to a [mini]batch
    index_history = T.lvector()     # index for history
    v = T.matrix('v')  # the data is presented as rasterized images
    p = T.matrix('p')  # the data is presented as rasterized images

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # construct the RBM class
    rbm = RBM_temporal_bin(input=v,
                           past=p,
                           piano_mapping=piano_mapping,
                           orch_mapping=orch_mapping,
                           n_visible=orch_dim,
                           n_hidden=n_hidden,
                           n_past=n_past,
                           np_rng=rng,
                           theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.cost_updates(lr=learning_rate, k=10)
    precision, recall, accuracy, updates_test = rbm.prediction_measure(gibbs_sampling_step_test)

    #################################
    #     Training the CRBM         #
    #################################

    # the purpose of train_crbm is solely to update the CRBM parameters
    train_temp_rbm = theano.function(inputs=[index, index_history],
                                     outputs=cost,
                                     updates=updates,
                                     givens={v: orch[index],
                                             p: create_past_vector(piano[index],
                                                                   orch[index_history],
                                                                   batch_size,
                                                                   temporal_order,
                                                                   orch_dim)},
                                     name='train_temp_rbm')

    train_temp_rbm_last_batch = theano.function(inputs=[index, index_history],
                                                outputs=cost,
                                                updates=updates,
                                                givens={v: orch[index],
                                                        p: create_past_vector(piano[index],
                                                                              orch[index_history],
                                                                              last_batch_size,
                                                                              temporal_order,
                                                                              orch_dim)},
                                                name='train_temp_rbm_last_batch')

    validation_error = theano.function(inputs=[index, index_history],
                                       outputs=[precision, recall, accuracy],
                                       updates=updates_test,
                                       givens={v: orch[index],
                                               p: create_past_vector(piano[index],
                                                                     orch[index_history],
                                                                     val_size,
                                                                     temporal_order,
                                                                     orch_dim)},
                                       name='validation_error')

    # Training step
    epoch = 0
    OVERFITTING = False
    val_order = 3
    val_tab = np.zeros(val_order)
    while (not OVERFITTING):
        # go through the training set
        train_cost_epoch = []
        for batch_index in xrange(n_train_batches - 1):
            # History indices
            hist_idx = np.array([train_index[batch_index] - n for n in xrange(1, temporal_order + 1)]).T
            # Train
            this_cost = train_temp_rbm(train_index[batch_index], hist_idx.ravel())
            # Keep track of cost
            train_cost_epoch += [this_cost]

        # Train last batch
        batch_index = n_train_batches - 1
        hist_idx = np.array([train_index[batch_index] - n for n in xrange(1, temporal_order + 1)]).T
        this_cost = train_temp_rbm_last_batch(train_index[batch_index], hist_idx.ravel())
        train_cost_epoch += [this_cost]

        if (epoch % 5 == 0):
            # Validation
            all_val_idx = []
            for i in xrange(0, len(val_index)):
                all_val_idx.extend(val_index[i])
            all_val_idx = np.array(all_val_idx)     # Oui, c'est dégueulasse, mais vraiment
            all_val_hist_idx = np.array([all_val_idx - n for n in xrange(1, temporal_order + 1)]).T
            _, _, accuracy = validation_error(all_val_idx, all_val_hist_idx.ravel())

            # Stop if validation error decreased over the last three validation
            # "FIFO" from the left
            val_tab[1:] = val_tab[0:-1]
            mean_accuracy = 100 * np.mean(accuracy)
            check_increase = np.sum(mean_accuracy > val_tab[1:])
            if check_increase == 0:
                OVERFITTING = True
            val_tab[0] = mean_accuracy
            # Monitor learning
            with open(log_file_path, 'wb') as log_file:
                log_file.write(("Epoch : {} , Rec error : {} , Valid acc : {}"
                               .format(epoch, np.mean(train_cost_epoch), mean_accuracy))
                               .encode('utf8'))
            print(("Epoch : {} , Rec error : {} , Valid acc : {}"
                  .format(epoch, np.mean(train_cost_epoch), mean_accuracy))
                  .encode('utf8'))

        epoch += 1

    return val_tab[-1]


def create_past_vector(piano, orch, batch_size, delay, orch_dim):
    # Piano is a matrix : num_batch x piano_dim
    # Orch a matrix : num_batch x ()
    # import sys
    # debug = sys.gettrace() is not None
    # import pdb; pdb.set_trace()
    # if debug:
    #     # Reshape checked, and has the expected behavior
    #     piano_dim = 5
    #     orch_dim = 10
    #     delay = 2
    #     batch_size = 3
    #     piano.tag.test_value = np.random.rand(batch_size, piano_dim)
    #     orch.tag.test_value = np.random.rand(batch_size * delay, orch_dim)
    orch_reshape = T.reshape(orch, (batch_size, delay * orch_dim))
    past = T.concatenate((piano, orch_reshape), axis=1)
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
