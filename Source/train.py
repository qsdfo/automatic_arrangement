#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

import sys
import time
import logging
import numpy as np
import cPickle as pkl
import os
# Perso
from Database.load_data import load_data_train, load_data_valid, load_data_test
from acidano.utils.early_stopping import up_criterion

import sys 
sys.setrecursionlimit(50000)

####################
# Debugging compiler flags
import theano
# theano.config.optimizer = 'fast_compile'
# theano.config.mode = 'FAST_COMPILE'
# theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'off'


def run_wrapper(params, config_folder, start_time_train):
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
    if script_param['model_class'] == "random":
        from acidano.models.lop.binary.random import Random as Model_class
    elif script_param['model_class'] == "repeat":
        from acidano.models.lop.binary.repeat import Repeat as Model_class
    elif script_param['model_class'] == "RBM":
        from acidano.models.lop.binary.RBM import RBM as Model_class
    elif script_param['model_class'] == "cRBM":
        from acidano.models.lop.binary.cRBM import cRBM as Model_class
    elif script_param['model_class'] == "FGcRBM":
        from acidano.models.lop.binary.FGcRBM import FGcRBM as Model_class
    elif script_param['model_class'] == "FGgru":
        from acidano.models.lop.binary.FGgru import FGgru as Model_class
    elif script_param['model_class'] == "FGcRBM_no_bv":
        from acidano.models.lop.binary.FGcRBM_no_bv import FGcRBM_no_bv as Model_class
    elif script_param['model_class'] == "FGcRnnRbm":
        from acidano.models.lop.binary.FGcRnnRbm import FGcRnnRbm as Model_class
    elif script_param['model_class'] == "LSTM":
        from acidano.models.lop.binary.LSTM import LSTM as Model_class
    elif script_param['model_class'] == "RnnRbm":
        from acidano.models.lop.binary.RnnRbm import RnnRbm as Model_class
    elif script_param['model_class'] == "cRnnRbm":
        from acidano.models.lop.binary.cRnnRbm import cRnnRbm as Model_class
    elif script_param['model_class'] == "cLstmRbm":
        from acidano.models.lop.binary.cLstmRbm import cLstmRbm as Model_class
    elif script_param['model_class'] == "FGcLstmRbm":
        from acidano.models.lop.binary.FGcLstmRbm import FGcLstmRbm as Model_class
    elif script_param['model_class'] == "LstmRbm":
        from acidano.models.lop.binary.LstmRbm import LstmRbm as Model_class
    elif script_param['model_class'] == "LSTM_gaussian_mixture":
        from acidano.models.lop.real.LSTM_gaussian_mixture import LSTM_gaussian_mixture as Model_class
    elif script_param['model_class'] == "LSTM_gaussian_mixture_2":
        from acidano.models.lop.real.LSTM_gaussian_mixture_2 import LSTM_gaussian_mixture_2 as Model_class

    if script_param['optimization_method'] == "gradient_descent":
        from acidano.utils.optim.gradient_descent import Gradient_descent as Optimization_method
    elif script_param['optimization_method'] == 'adam_L2':
        from acidano.utils.optim.adam_L2 import Adam_L2 as Optimization_method
    elif script_param['optimization_method'] == 'rmsprop':
        from acidano.utils.optim.rmsprop import Rmsprop as Optimization_method
    elif script_param['optimization_method'] == 'sgd_nesterov':
        from acidano.utils.optim.sgd_nesterov import Sgd_nesterov as Optimization_method

    ############################################################
    # Logging
    ############################################################
    log_file_path = config_folder + '/' + 'log.txt'
    with open(log_file_path, 'wb') as f:
        f.close()
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger_run = logging.getLogger('run')
    hdlr = logging.FileHandler(log_file_path)
    hdlr.setFormatter(formatter)
    logger_run.addHandler(hdlr)

    ############################################################
    # Load data
    ############################################################
    time_load_0 = time.time()
    piano_train, orchestra_train, train_index \
        = load_data_train(script_param['data_folder'],
                          None, None,
                          model_param['temporal_order'],
                          model_param['batch_size'],
                          skip_sample=script_param['skip_sample'],
                          avoid_silence=script_param['avoid_silence'],
                          binarize_piano=script_param['binarize_piano'],
                          binarize_orchestra=script_param['binarize_orchestra'],
                          logger_load=logger_run)
    piano_valid, orchestra_valid, valid_index \
        = load_data_valid(script_param['data_folder'],
                          None, None,
                          model_param['temporal_order'],
                          model_param['batch_size'],
                          skip_sample=script_param['skip_sample'],
                          avoid_silence=True,
                          binarize_piano=script_param['binarize_piano'],
                          binarize_orchestra=script_param['binarize_orchestra'],
                          logger_load=logger_run)
    # This load is only for sanity check purposes
    piano_test, orchestra_test, _, _ \
        = load_data_test(script_param['data_folder'],
                         None, None,
                         model_param['temporal_order'],
                         model_param['batch_size'],
                         skip_sample=script_param['skip_sample'],
                         avoid_silence=True,
                         binarize_piano=script_param['binarize_piano'],
                         binarize_orchestra=script_param['binarize_orchestra'],
                         logger_load=logger_run)
    time_load_1 = time.time()
    logger_run.info('TTT : Loading data took {} seconds'.format(time_load_1-time_load_0))
    # Create the checksum dictionnary
    checksum_database = {
        'piano_train': piano_train.sum(),
        'orchestra_train': orchestra_train.sum(),
        'piano_valid': piano_valid.sum(),
        'orchestra_valid': orchestra_valid.sum(),
        'piano_test': piano_test.sum(),
        'orchestra_test': orchestra_test.sum()
    }

    ############################################################
    # Get dimensions of batches
    ############################################################
    piano_dim = piano_train.shape[1]
    orchestra_dim = orchestra_train.shape[1]
    dimensions = {'batch_size': model_param['batch_size'],
                  'temporal_order': model_param['temporal_order'],
                  'piano_dim': piano_dim,
                  'orchestra_dim': orchestra_dim}

    ############################################################
    # Update train_param and model_param dicts with new information from load data
    ############################################################
    n_train_batches = len(train_index)
    n_val_batches = len(valid_index)

    logger_run.info((u'##### Data').encode('utf8'))
    logger_run.info((u'# n_train_batch :  {}'.format(n_train_batches)).encode('utf8'))
    logger_run.info((u'# n_val_batch :  {}'.format(n_val_batches)).encode('utf8'))

    train_param['n_train_batches'] = n_train_batches
    train_param['n_val_batches'] = n_val_batches
    train_param['start_time_train'] = start_time_train

    # Class normalization
    notes_activation = orchestra_train.sum(axis=0)
    notes_activation_norm = notes_activation.mean() / (notes_activation+1e-10)
    class_normalization = np.maximum(1, np.minimum(20, notes_activation_norm))
    model_param['class_normalization'] = class_normalization

    # Other kind of regularization
    L_train = orchestra_train.shape[0]
    mean_notes_activation = notes_activation / L_train
    mean_notes_activation = np.where(mean_notes_activation == 0, 1. / L_train, mean_notes_activation)
    model_param['mean_notes_activation'] = mean_notes_activation

    ############################################################
    # Instanciate model and Optimization method
    ############################################################
    logger_run.info((u'##### Model parameters').encode('utf8'))
    for k, v in model_param.iteritems():
        logger_run.info((u'# ' + k + ' :  {}'.format(v)).encode('utf8'))
    logger_run.info((u'##### Optimization parameters').encode('utf8'))
    for k, v in optim_param.iteritems():
        logger_run.info((u'# ' + k + ' :  {}'.format(v)).encode('utf8'))

    model = Model_class(model_param, dimensions, checksum_database)

    # Define an optimizer
    optimizer = Optimization_method(optim_param)

    ############################################################
    # Train
    ############################################################
    time_train_0 = time.time()
    loss, accuracy, best_epoch, best_model = train(model, optimizer,
                                    piano_train, orchestra_train, train_index,
                                    piano_valid, orchestra_valid, valid_index,
                                    train_param, config_folder, logger_run)

    time_train_1 = time.time()
    training_time = time_train_1-time_train_0
    logger_run.info('TTT : Training data took {} seconds'.format(training_time))
    logger_run.info((u'# Best model obtained at epoch :  {}'.format(best_epoch)).encode('utf8'))
    logger_run.info((u'# Accuracy :  {}'.format(accuracy)).encode('utf8'))
    logger_run.info((u'###################\n').encode('utf8'))

    ############################################################
    # Pickle (save) the model for plotting weights
    ############################################################
    save_model_file = config_folder + '/model.pkl'
    with open(save_model_file, 'wb') as f:
        pkl.dump(best_model, f, protocol=pkl.HIGHEST_PROTOCOL)

    ############################################################
    # Write result in a txt file
    ############################################################
    result_file_path = config_folder + '/result.csv'
    with open(result_file_path, 'wb') as f:
        f.write("accuracy;" + str(accuracy) + '\n' + "loss;" + str(loss))

    # Close handler
    logger_run.removeHandler(hdlr)
    return


def train(model, optimizer,
          piano_train, orchestra_train, train_index,
          piano_valid, orchestra_valid, valid_index,
          train_param, config_folder, logger_train):
    ############################################################
    # Time information used
    ############################################################
    time_limit = train_param['walltime'] * 3600 - 30*60  # walltime - 30 minutes in seconds
    start_time_train = train_param['start_time_train']

    ############################################################
    # Compile theano functions
    # Compilation of the training function is encapsulated in the class since the 'givens'
    # can vary with the model
    ############################################################
    model.build_train_fn(optimizer, name='train_iteration')
    # Same for the validation
    model.build_validation_fn(name='validation_error')

    ############################################################
    # Training
    ############################################################
    logger_train.info("#")
    logger_train.info("# Training")
    epoch = 0
    OVERFITTING = False
    TIME_LIMIT = False
    val_tab = np.zeros(max(1, train_param['max_iter']))
    loss_tab = np.zeros(max(1, train_param['max_iter']))
    best_model = None
    best_epoch = None
    while (not OVERFITTING and not TIME_LIMIT
           and epoch != train_param['max_iter']):

        start_time_epoch = time.time()

        #######################################
        # Train
        #######################################
        train_cost_epoch = []
        train_monitor_epoch = []
        ###### # # # # ## # ## ## #  #
        # ind_activation = np.random.randint(model.batch_size, size=(train_param['n_train_batches']))
        # random_choice_mean_activation = np.zeros((model.k, model.n_v, train_param['n_train_batches']))
        # mean_activation = np.zeros((model.k, model.n_v, train_param['n_train_batches']))
        ###### # # # # ## # ## ## #  #
        for batch_index in train_index:
            batch_data = model.generator(piano_train, orchestra_train, batch_index)
            this_cost, this_monitor = model.train_batch(batch_data)
            # Keep track of cost
            train_cost_epoch.append(this_cost)
            train_monitor_epoch.append(this_monitor)
            # # Plot a random mean_chain
            # random_choice_mean_activation[:, :, batch_index] = mean_chain[:, ind_activation[batch_index], :]
            # # mean along batch axis
            # mean_activation[:, :, batch_index] = mean_chain.mean(axis=1)

        mean_loss = np.mean(train_cost_epoch)
        loss_tab[epoch] = mean_loss

        #######################################
        # Validate
        # For binary unit, it's an accuracy measure.
        # For real valued units its a gaussian centered value with variance 1
        #######################################
        accuracy = []
        for batch_index in valid_index:
            batch_data = model.generator(piano_valid, orchestra_valid, batch_index)
            # _, _, accuracy_batch, true_frame, past_frame, piano_frame, predicted_frame = validation_error(valid_index[batch_index])
            _, _, accuracy_batch = model.validate_batch(batch_data)
            accuracy += [accuracy_batch]

            # if train_param['DEBUG']:
                # from acidano.visualization.numpy_array.visualize_numpy import visualize_mat
                # if batch_index == 0:
                #     for ind in range(accuracy_batch.shape[0]):
                #         pr_viz = np.zeros((4, predicted_frame.shape[1]))
                #         # Threshold prediction
                #         orch_pred_ind = predicted_frame[ind]
                #         # Less than 1%
                #         thresh_pred = np.where(orch_pred_ind > 0.01, orch_pred_ind, 0)
                #         pr_viz[0] = thresh_pred
                #         pr_viz[1] = true_frame[ind]
                #         pr_viz[2] = past_frame[ind]
                #         pr_viz[3][:piano_frame.shape[1]] = piano_frame[ind]
                #         path_accuracy = config_folder + '/DEBUG/' + str(epoch) + '/validation'
                #         if not os.path.isdir(path_accuracy):
                #             os.makedirs(path_accuracy)
                #         visualize_mat(np.transpose(pr_viz), path_accuracy, str(ind) + '_score_' + str(accuracy_batch[ind]))
        mean_accuracy = 100 * np.mean(accuracy)

        end_time_epoch = time.time()

        #######################################
        # Is it the best model we have seen so far ?
        if mean_accuracy >= np.max(val_tab):
            best_model = model
            best_epoch = epoch
        #######################################

        #######################################
        # DEBUG : plot weights
        #######################################
        if train_param['DEBUG']:
            from acidano.visualization.numpy_array.visualize_numpy import visualize_mat_proba
            
            # Visible activations
            # path_activation = config_folder + '/DEBUG/' + str(epoch) + '/activations'
            # if not os.path.isdir(path_activation):
            #     os.makedirs(path_activation)
            # mean_activation = mean_activation.mean(axis=2)
            # visualize_mat_proba(mean_activation, path_activation, 'mean_activations')
            # Do not plot every random activation...
            # for i in np.linspace(0, random_choice_mean_activation.shape[2], 10, endpoint=False):
            #     ind = int(i)
            #     visualize_mat_proba(random_choice_mean_activation[:, :, ind], path_activation, 'random_act_' + str(ind))
            
            # Weights
            # plot_folder = config_folder + '/DEBUG/' + str(epoch) + '/weights'
            # if not os.path.isdir(plot_folder):
            #     os.makedirs(plot_folder)
            # model.save_weights(plot_folder) 

        #######################################
        # OLD VERSION
        # Early stopping criterion
        # Note that sum_{i=0}^{n} der = der(n) - der(0)
        # So mean over successive derivatives makes no sense
        # 1/ Get the mean derivative between 5 and 10 =
        #       \sum_{i=validation_order}^{validation_order+initial_derivative_length} E(i) - E(i-validation_order) / validation_order
        #
        # 2/ At each iteration, compare the mean derivative over the last five epochs :
        #       \sum_{i=0}^{validation_order} E(t)
        #
        # val_tab[epoch] = mean_accuracy
        # if epoch == train_param['initial_derivative_length']-1:
        #     ind = np.arange(train_param['validation_order']-1, train_param['initial_derivative_length'])
        #     increase_reference = (val_tab[ind] - val_tab[ind-train_param['validation_order']+1]).sum() / (train_param['validation_order'] * len(ind))
        #     if increase_reference <= 0:
        #         # Early stop if the model didn't really improved over the first iteration
        #         DIVERGING = True
        # elif epoch >= train_param['initial_derivative_length']:
        #     ind = np.arange(epoch - train_param['check_derivative_length'] + 1, epoch+1)
        #     derivative_mean = (val_tab[ind] - val_tab[ind-train_param['validation_order']+1]).sum() / (train_param['validation_order'] * len(ind))
        #     # Mean derivative is less than 10% of increase reference
        #     if derivative_mean < 0.1 * increase_reference:
        #         OVERFITTING = True
        #######################################

        #######################################
        # Article
        # Early stopping, but when ?
        # Lutz Prechelt
        # UP criterion (except that we expect accuracy to go up in our case,
        # so the minus sign)
        val_tab[epoch] = mean_accuracy
        if epoch >= train_param['min_number_iteration']:
            OVERFITTING = up_criterion(-val_tab, epoch, train_param["number_strips"], train_param["validation_order"])
        #######################################

        #######################################
        # Monitor time (guillimin walltime)
        if (time.time() - start_time_train) > time_limit:
            TIME_LIMIT = True
        #######################################

        #######################################
        # Log training
        #######################################
        logger_train.info(('Epoch : {} , Monitor : {} , Cost : {} , Valid acc : {}'
                          .format(epoch, np.mean(train_monitor_epoch), mean_loss, mean_accuracy))
                          .encode('utf8'))

        logger_train.info(('Time : {}'
                          .format(end_time_epoch - start_time_epoch))
                          .encode('utf8'))

        if OVERFITTING:
            logger_train.info('OVERFITTING !!')

        if TIME_LIMIT:
            logger_train.info('TIME OUT !!')

        #######################################
        # Epoch +1
        #######################################
        epoch += 1

    # Return best accuracy
    best_accuracy = val_tab[best_epoch]
    best_loss = loss_tab[best_epoch]
    return best_loss, best_accuracy, best_epoch, best_model


if __name__ == '__main__':
    start_time_train = time.time()

    config_folder = sys.argv[1]
    params = pkl.load(open(config_folder + '/config.pkl', "rb"))
    run_wrapper(params, config_folder, start_time_train)

    #####################################################
    ##### Retrain an existing config
    # config_folder = "DEBUG/FGcRBM_weight_decay"
    # #### Perhaps you need to change some paths variables
    # params = pkl.load(open(config_folder + '/config.pkl', "rb"))
    # params['model']['weight_decay_coeff'] = 10
    # params['script']['result_folder'] = config_folder
    # params['script']['data_folder'] = "DEBUG/FGcRBM_weight_decay/Data"
    # params['train']['walltime'] = 16
    # params['train']['max_iter'] = 3
    # import pdb; pdb.set_trace()
    # params['train']['DEBUG'] = True
    # pkl.dump(params, open(config_folder + '/config.pkl', "wb"))
    # run_wrapper(params, config_folder, start_time_train)
    #####################################################
    #####################################################
