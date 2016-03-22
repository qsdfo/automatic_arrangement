#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
Hyperopt
RNN-RBM w/ binary units
"""
# Numpy
import numpy as np
# Theano
import theano
import theano.tensor as T
# Hyperopt
from hyperopt import hp
from math import log

from Data_processing.load_data import load_data_seq_tvt
from Models.RnnRbm.class_def import RnnRbm


# Define hyper-parameter search space
def get_header():
    return ['n_hidden', 'n_hidden_recurrent', 'temporal_order',
            'learning_rate_RBM', 'learning_rate_RNN', 'K', 'accuracy']


def get_hp_space():
    space = (hp.qloguniform('n_hidden', log(100), log(5000), 10),
             hp.qloguniform('n_hidden_recurrent', log(100), log(5000), 1),
             hp.qloguniform('temporal_order', log(10), log(100), 10),
             hp.loguniform('learning_rate_RBM', log(0.0001), log(1)),
             hp.loguniform('learning_rate_RNN', log(0.0001), log(1)),
             hp.qloguniform('K', log(2), log(20), 1)
             # Different learninf rate for RBM and RNN
             # hp.choice('activation_func', ['tanh', 'sigmoid']),
             # hp.choice('sampling_positive', ['true', 'false'])
             # gibbs_sampling_step_test ???
             )
    return space


def train(params, dataset, temporal_granularity, log_file_path):
    # Hyperparams
    # n_hidden, n_hidden_recurrent, learning_rate, activation_func, sampling_positive = params
    n_hidden, n_hidden_recurrent, temporal_order, learning_rate_RBM, learning_rate_RNN, K = params

    # Cast the hp
    n_hidden = int(n_hidden)
    n_hidden_recurrent = int(n_hidden_recurrent)
    temporal_order = int(temporal_order)
    K = int(K)

    # Log them
    with open(log_file_path, 'ab') as log_file:
        log_file.write((u'# n_hidden :  {}\n'.format(n_hidden)).encode('utf8'))
        log_file.write((u'# n_hidden_recurrent :  {}\n'.format(n_hidden_recurrent)).encode('utf8'))
        log_file.write((u'# temporal_order :  {}\n'.format(temporal_order)).encode('utf8'))
        log_file.write((u'# learning_rate_RBM :  {}\n'.format(learning_rate_RBM)).encode('utf8'))
        log_file.write((u'# learning_rate_RNN :  {}\n'.format(learning_rate_RNN)).encode('utf8'))
        log_file.write((u'# K :  {}\n'.format(K)).encode('utf8'))
    # Print
    print((u'# n_hidden :  {}'.format(n_hidden)).encode('utf8'))
    print((u'# n_hidden_recurrent :  {}'.format(n_hidden_recurrent)).encode('utf8'))
    print((u'# temporal_order :  {}'.format(temporal_order)).encode('utf8'))
    print((u'# learning_rate_RBM :  {}'.format(learning_rate_RBM)).encode('utf8'))
    print((u'# learning_rate_RNN :  {}'.format(learning_rate_RNN)).encode('utf8'))
    print((u'# K :  {}'.format(K)).encode('utf8'))

    # Some hp not optimized
    gibbs_sampling_step_test = 40

    # Load data
    # Dimension : time * pitch
    orch, orch_mapping, piano, piano_mapping, train_index, val_index, _ \
        = load_data_seq_tvt(data_path=dataset,
                            log_file_path='bullshit.txt',
                            temporal_granularity=temporal_granularity,
                            temporal_order=temporal_order,
                            shared_bool=True,
                            bin_unit_bool=True,
                            split=(0.7, 0.1, 0.2))

    # Get dimensions
    orch_dim = orch.get_value(borrow=True).shape[1]
    piano_dim = piano.get_value(borrow=True).shape[1]

    # allocate symbolic variables for the data
    index = T.lvector()             # index to a [mini]batch
    o = T.matrix('o')
    p = T.matrix('p')

    # construct the RnnRbm class
    model = RnnRbm(orch=o,          # sequences as Theano matrices
                   piano=p,         # sequences as Theano matrices
                   n_orch=orch_dim,
                   n_piano=piano_dim,
                   n_hidden=n_hidden,
                   n_hidden_recurrent=n_hidden_recurrent,
                   )

    # get the cost and the gradient corresponding to one step of CD-15
    cost, monitor, updates = model.cost_updates(lr_RBM=learning_rate_RBM,
                                                lr_RNN=learning_rate_RNN,
                                                k=K)
    precision, recall, accuracy, updates_test = model.prediction_measure(k=gibbs_sampling_step_test)

    #################################
    #     Training the CRBM         #
    #################################

    # the purpose of train_crbm is solely to update the CRBM parameters
    train_rnnrbm = theano.function(inputs=[index],
                                   outputs=[cost, monitor],
                                   updates=updates,
                                   givens={o: orch[index],
                                           p: piano[index]},
                                   name='train_rnnrbm')

    validation_error = theano.function(inputs=[index],
                                       outputs=[precision, recall, accuracy],
                                       updates=updates_test,
                                       givens={o: orch[index],
                                               p: piano[index]},
                                       name='validation_error')

    # Training step
    epoch = 0
    OVERFITTING = False
    val_order = 4
    val_tab = np.zeros(val_order)
    while (not OVERFITTING):
        # go through the training set
        train_cost_epoch = []
        for ind_batch in train_index:
            # Train
            this_cost, this_monitor = train_rnnrbm(ind_batch)
            # Keep track of MONITORING cost
            train_cost_epoch += [this_monitor]

        if (epoch % 5 == 0):
            # Validation
            acc_store = []
            for ind_batch in val_index:
                _, _, acc = validation_error(ind_batch)
                acc_store += [acc]

            # Stop if validation error decreased over the last three validation
            # "FIFO" from the left
            val_tab[1:] = val_tab[0:-1]
            mean_accuracy = 100 * np.mean(acc_store)
            check_increase = np.sum(mean_accuracy >= val_tab[1:])
            if check_increase == 0:
                OVERFITTING = True
            val_tab[0] = mean_accuracy
            # Monitor learning
            with open(log_file_path, 'ab') as log_file:
                log_file.write(("Epoch : {} , Rec error : {} , Valid acc : {}"
                               .format(epoch, np.mean(train_cost_epoch), mean_accuracy))
                               .encode('utf8'))
            print(("Epoch : {} , Rec error : {} , Valid acc : {}"
                  .format(epoch, np.mean(train_cost_epoch), mean_accuracy))
                  .encode('utf8'))

        epoch += 1

    best_accuracy = np.amax(val_tab)
    dico_res = {'n_hidden': n_hidden,
                'n_hidden_recurrent': n_hidden_recurrent,
                'temporal_order': temporal_order,
                'learning_rate_RBM': learning_rate_RBM,
                'learning_rate_RNN': learning_rate_RNN,
                'K': K,
                'accuracy': best_accuracy}

    return best_accuracy, dico_res


def create_past_vector(piano, orch, batch_size, delay, orch_dim):
    orch_reshape = T.reshape(orch, (batch_size, delay * orch_dim))
    past = T.concatenate((piano, orch_reshape), axis=1)
    return past
