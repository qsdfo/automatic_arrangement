#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import time
import logging
import numpy as np
import cPickle as pkl
# Perso
from load_data import load_data
# hopt
from hyperopt import STATUS_OK

####################
# Debugging compiler flags
import theano
# theano.config.optimizer = 'fast_compile'
# theano.config.mode = 'FAST_COMPILE'
# theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'off'


def run_wrapper(params, config_folder):
    ############################################################
    # Unpack parameters
    ############################################################
    script_param = params['script']
    train_param = params['train']
    model_param = params['model']
    optim_param = params['optim']

    ############################################################
    # Load models and optim modules
    ############################################################
    if script_param['model_class'] == "RBM":
        from acidano.models.lop.discrete.RBM import RBM as Model_class
    elif script_param['model_class'] == "cRBM":
        from acidano.models.lop.discrete.cRBM import cRBM as Model_class
    elif script_param['model_class'] == "FGcRBM":
        from acidano.models.lop.discrete.FGcRBM import FGcRBM as Model_class
    elif script_param['model_class'] == "LSTM":
        from acidano.models.lop.discrete.LSTM import LSTM as Model_class
    elif script_param['model_class'] == "RnnRbm":
        from acidano.models.lop.discrete.RnnRbm import RnnRbm as Model_class
    elif script_param['model_class'] == "cRnnRbm":
        from acidano.models.lop.discrete.cRnnRbm import cRnnRbm as Model_class
    elif script_param['model_class'] == "LSTM_gaussian_mixture":
        from acidano.models.lop.real.LSTM_gaussian_mixture import LSTM_gaussian_mixture as Model_class
    elif script_param['model_class'] == "LSTM_gaussian_mixture_2":
        from acidano.models.lop.real.LSTM_gaussian_mixture_2 import LSTM_gaussian_mixture_2 as Model_class

    if script_param['optimization_method'] == "gradient_descent":
        from acidano.utils.optim import gradient_descent as Optimization_method
    elif script_param['optimization_method'] == 'adam_L2':
        from acidano.utils.optim import adam_L2 as Optimization_method
    elif script_param['optimization_method'] == 'rmsprop':
        from acidano.utils.optim import rmsprop as Optimization_method
    elif script_param['optimization_method'] == 'sgd_nesterov':
        from acidano.utils.optim import sgd_nesterov as Optimization_method

    ############################################################
    # Paths
    ############################################################
    log_file_path = config_folder + '/' + 'log'

    ############################################################
    # Logging
    ############################################################
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_file_path,
                        filemode='w')

    logger_run = logging.getLogger('run')
    logger_run.info(('\n').encode('utf8'))
    logger_run.info((u'#'*40).encode('utf8'))

    ############################################################
    # Load data
    ############################################################
    time_load_0 = time.time()
    piano_train, orchestra_train, train_index, \
        piano_valid, orchestra_valid, valid_index, \
        piano_test, orchestra_test, test_index, generation_index \
        = load_data(script_param['data_folder'],
                    model_param['temporal_order'],
                    model_param['batch_size'],
                    binary_unit=script_param['binary_unit'],
                    skip_sample=1,
                    logger_load=logger_run)
    time_load_1 = time.time()
    logger_run.info('TTT : Loading data took {} seconds'.format(time_load_1-time_load_0))

    ############################################################
    # Get dimensions of batches
    ############################################################
    piano_dim = piano_train.get_value().shape[1]
    orchestra_dim = orchestra_train.get_value().shape[1]
    dimensions = {'batch_size': model_param['batch_size'],
                  'temporal_order': model_param['temporal_order'],
                  'piano_dim': piano_dim,
                  'orchestra_dim': orchestra_dim}

    ############################################################
    # Update train_param dict with new information from load data
    ############################################################
    n_train_batches = len(train_index)
    n_val_batches = len(valid_index)

    logger_run.info((u'##### Data').encode('utf8'))
    logger_run.info((u'# n_train_batch :  {}'.format(n_train_batches)).encode('utf8'))
    logger_run.info((u'# n_val_batch :  {}'.format(n_val_batches)).encode('utf8'))

    train_param['n_train_batches'] = n_train_batches
    train_param['n_val_batches'] = n_val_batches

    ############################################################
    # Instanciate model and Optimization method
    ############################################################
    logger_run.info((u'##### Model parameters').encode('utf8'))
    for k, v in model_param.iteritems():
        logger_run.info((u'# ' + k + ' :  {}'.format(v)).encode('utf8'))
    logger_run.info((u'##### Optimization parameters').encode('utf8'))
    for k, v in optim_param.iteritems():
        logger_run.info((u'# ' + k + ' :  {}'.format(v)).encode('utf8'))

    model = Model_class(model_param, dimensions)
    # Define an optimizer
    optimizer = Optimization_method(optim_param)

    ############################################################
    # Train
    ############################################################
    time_train_0 = time.time()
    loss, accuracy = train(model, optimizer,
                           piano_train, orchestra_train, train_index,
                           piano_valid, orchestra_valid, valid_index,
                           train_param, logger_run)
    time_train_1 = time.time()
    training_time = time_train_1-time_train_0
    logger_run.info('TTT : Training data took {} seconds'.format(training_time))
    logger_run.info((u'# Accuracy :  {}'.format(accuracy)).encode('utf8'))
    logger_run.info((u'###################\n').encode('utf8'))

    ############################################################
    # Pickle (save) the model for plotting weights
    ############################################################
    save_model_file = config_folder + '/model.pkl'
    with open(save_model_file, 'wb') as f:
        pkl.dump(model, f, protocol=pkl.HIGHEST_PROTOCOL)

    ############################################################
    # Write result in a txt file
    ############################################################
    result_file_path = config_folder + '/result.csv'
    with open(result_file_path, 'wb') as f:
        f.write("accuracy;" + str(accuracy) + '\n' + "loss;" + str(loss))

    return


def train(model, optimizer,
          piano_train, orchestra_train, train_index,
          piano_valid, orchestra_valid, valid_index,
          train_param, logger_train):
    ############################################################
    # Compile theano functions
    # Compilation of the training function is encapsulated in the class since the 'givens'
    # can vary with the model
    ############################################################
    train_iteration = model.get_train_function(piano_train, orchestra_train, optimizer, name='train_iteration')
    # Same for the validation
    validation_error = model.get_validation_error(piano_valid, orchestra_valid, name='validation_error')

    ############################################################
    # Training
    ############################################################
    logger_train.info("#")
    logger_train.info("# Training")
    epoch = 0
    OVERFITTING = False
    DIVERGING = False
    val_tab = np.zeros(max(1,train_param['max_iter']))
    loss_tab = np.zeros(max(1,train_param['max_iter']))
    while (not OVERFITTING
           and not DIVERGING
           and epoch!=train_param['max_iter']):
        #######################################
        # Train
        #######################################
        train_cost_epoch = []
        train_monitor_epoch = []
        for batch_index in xrange(train_param['n_train_batches']):
            this_cost, this_monitor = train_iteration(train_index[batch_index])
            # Keep track of cost
            train_cost_epoch.append(this_cost)
            train_monitor_epoch.append(this_monitor)

        mean_loss = np.mean(train_cost_epoch)
        loss_tab[epoch] = mean_loss

        #######################################
        # Validate
        # For binary unit, it's an accuracy measure.
        # For real valued units its a gaussian centered value with variance 1
        #######################################
        accuracy = []
        for batch_index in xrange(train_param['n_val_batches']):
            _, _, accuracy_batch = validation_error(valid_index[batch_index])
            accuracy += [accuracy_batch]
        mean_accuracy = 100 * np.mean(accuracy)

        #######################################
        # Early stopping criterion
        # Note that sum_{i=0}^{n} der = der(n) - der(0)
        # So mean over successive derivatives makes no sense
        # 1/ Get the mean derivative between 5 and 10 =
        #       \sum_{i=validation_order}^{validation_order+initial_derivative_length} E(i) - E(i-validation_order) / validation_order
        #
        # 2/ At each iteration, compare the mean derivative over the last five epochs :
        #       \sum_{i=0}^{validation_order} E(t)
        #######################################
        val_tab[epoch] = mean_accuracy
        if epoch == train_param['initial_derivative_length']-1:
            ind = np.arange(train_param['validation_order']-1, train_param['initial_derivative_length'])
            increase_reference = (val_tab[ind] - val_tab[ind-train_param['validation_order']+1]).sum() / (train_param['validation_order'] * len(ind))
            if increase_reference <= 0:
                # Early stop if the model didn't really improved over the first iteration
                DIVERGING = True
        elif epoch >= train_param['initial_derivative_length']:
            ind = np.arange(epoch - train_param['check_derivative_length'] + 1, epoch+1)
            derivative_mean = (val_tab[ind] - val_tab[ind-train_param['validation_order']+1]).sum() / (train_param['validation_order'] * len(ind))
            # Mean derivative is less than 10% of increase reference
            if derivative_mean < 0.1 * increase_reference:
                OVERFITTING = True

        #######################################
        # Log training
        #######################################
        logger_train.info(('Epoch : {} , Monitor : {} , Cost : {} , Valid acc : {}'
                          .format(epoch, np.mean(train_monitor_epoch), mean_loss, mean_accuracy))
                          .encode('utf8'))
        if DIVERGING:
            logger_train.info('DIVERGING !!')
        elif OVERFITTING:
            logger_train.info('OVERFITTING !!')

        #######################################
        # Epoch +1
        #######################################
        epoch += 1

    best_epoch = np.argmax(val_tab)
    best_accuracy = val_tab[best_epoch]
    best_loss = loss_tab[best_epoch]
    return best_loss, best_accuracy

if __name__ == '__main__':
    config_folder = sys.argv[1]
    params = pkl.load(open(config_folder + '/config.pkl', "rb"))
    import pdb; pdb.set_trace()
    run_wrapper(params, config_folder)
