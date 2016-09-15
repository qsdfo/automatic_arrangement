#!/usr/bin/env python
# -*- coding: utf8 -*-

# Main script for music generation

import os
import csv
# Hyperopt
from hyperopt import fmin, tpe
# Logging
import logging
# Perso
from load_data import load_data
from build_data import build_data
import theano.tensor as T
import numpy as np
####################
# Reminder for plotting tools
# import matplotlib.pyplot as plt
# Histogram
# n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# plt.show()

####################
# Debugging compiler flags
# import theano
# theano.config.optimizer = 'None'
# theano.config.mode = 'FAST_COMPILE'
# theano.config.exception_verbosity = 'high'
# theano.config.compute_test_value = 'warn'

####################
# Select a model (path to the .py file)
# Two things define a model : it's architecture and the time granularity
from acidano.models.lop.cRBM import cRBM as Model_class
from acidano.utils.optim import gradient_descent as Optimization_method

# Build data parameters :
REBUILD_DATABASE = False
# Temporal granularity and quantization
temporal_granularity = u'event_level'
quantization = 4

# Get main dir
MAIN_DIR = os.getcwd().decode('utf8') + u'/'

# Set hyperparameters (can be a grid)
result_folder = MAIN_DIR + u'../Results/' + temporal_granularity + '/' + Model_class.name()
result_file = result_folder + u'/hopt_results.csv'
log_file_path = result_folder + '/' + Model_class.name() + u'.log'

# Fixed hyper parameter
max_evals = 20       # number of hyper-parameter configurations evaluated
max_iter = 200      # nb max of iterations when training 1 configuration of hparams
# Config is set now, no need to modify source below for standard use
############################################################################
############################################################################


def train_hopt(temporal_granularity, max_evals, log_file_path, csv_file_path):
    # Create/reinit csv file
    open(csv_file_path, 'w').close()

    logger_hopt.info((u'WITH HYPERPARAMETER OPTIMIZATION').encode('utf8'))
    logger_hopt.info((u'**** Model : ' + Model_class.name()).encode('utf8'))
    logger_hopt.info((u'**** Optimization technic : ' + Optimization_method.name()).encode('utf8'))
    logger_hopt.info((u'**** Temporal granularity : ' + temporal_granularity + '\n').encode('utf8'))

    # Define hyper-parameter search space for the model
    # Those are given by the static methods get_param_dico and get_hp_space
    param_space = Model_class.get_hp_space()
    optim_space = Optimization_method.get_hp_space()
    space = param_space + optim_space

    # Get the headers (i.e. list of hyperparameters tuned for printing and
    # save purposes)
    param_header = (Model_class.get_param_dico(None)).keys()
    optim_header = (Optimization_method.get_param_dico(None)).keys()
    header = param_header + optim_header + ['accuracy']

    global run_counter
    run_counter = 0

    def run_wrapper(params):
        global run_counter
        run_counter += 1
        logger_hopt.info(('\n').encode('utf8'))
        logger_hopt.info((u'#'*40).encode('utf8'))
        logger_hopt.info((u'# Config :  {}'.format(run_counter)).encode('utf8'))

        # Map hparam into a dictionary ##############
        num_model_param = len(param_header)
        model_param = Model_class.get_param_dico(params[:num_model_param])
        optim_param = Optimization_method.get_param_dico(params[num_model_param:])
        #############################################

        # Train #####################################
        dico_res = train(model_param, optim_param, max_iter, log_file_path)
        error = -dico_res['accuracy']  # Search for a min
        #############################################

        # log
        logger_hopt.info((u'# Accuracy :  {}'.format(dico_res['accuracy'])).encode('utf8'))
        logger_hopt.info((u'###################\n').encode('utf8'))

        # Write the result in result.csv
        with open(csv_file_path, 'ab') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=header)
            writer.writerow(dico_res)

        return error

    # Calling get_param_dico with None return an empty dictionary,
    # Useful to get the header of hparam
    with open(csv_file_path, 'ab') as csvfile:
        # Write headers if they don't already exist
        writerHead = csv.writer(csvfile, delimiter=',')
        writerHead.writerow(header)

    best = fmin(run_wrapper, space, algo=tpe.suggest, max_evals=max_evals)

    return best


def train(model_param, optim_param, max_iter, log_file_path):
    ############################################################
    ############################################################
    ############################################################
    # model_param and optim_param are dictionaries
    # If you use train directly, bypassing the hparameter loop,
    # be careful that the keys match the constructor arguments of both model and optimizer

    # Log them
    logger_train.info((u'##### Model parameters').encode('utf8'))
    for k, v in model_param.iteritems():
        logger_train.info((u'# ' + k + ' :  {}'.format(v)).encode('utf8'))
    logger_train.info((u'##### Optimization parameters').encode('utf8'))
    for k, v in optim_param.iteritems():
        logger_train.info((u'# ' + k + ' :  {}'.format(v)).encode('utf8'))

    ################################################################
    ################################################################
    ################################################################
    # DATA
    # Dimension : time * pitch
    piano_train, orchestra_train, train_index, \
        piano_valid, orchestra_valid, valid_index, \
        piano_test, orchestra_test, test_index \
        = load_data(model_param['temporal_order'],
                    model_param['batch_size'],
                    binary_unit=True,
                    skip_sample=1,
                    logger_load=logger_load)
    # For large datasets
    #   http://deeplearning.net/software/theano/tutorial/aliasing.html
    #   use borrow=True (avoid copying the whole matrix) ?
    #   Load as much as the GPU can handle, train then load other
    #       part of the dataset using shared_variable.set_value(new_value)
    piano_dim = piano_train.get_value().shape[1]
    orchestra_dim = orchestra_train.get_value().shape[1]
    n_train_batches = len(train_index)
    n_val_batches = len(valid_index)
    logger_train.info((u'# n_train_batch :  {}'.format(n_train_batches)).encode('utf8'))
    logger_train.info((u'# n_val_batch :  {}'.format(n_val_batches)).encode('utf8'))

    ################################################################
    ################################################################
    ################################################################
    # MODEL
    # dimensions dictionary
    dimensions = {'batch_size': model_param['batch_size'],
                  'temporal_order': model_param['temporal_order'],
                  'piano_dim': piano_dim,
                  'orchestra_dim': orchestra_dim}
    model = Model_class(model_param, dimensions)
    # Define an optimizer
    optimizer = Optimization_method(optim_param)

    ############################################################
    ############################################################
    ############################################################
    # COMPILE FUNCTIONS
    # index to a [mini]batch : int32
    index = T.ivector()
    # Compilation of the training function is encapsulated in the class since the 'givens'
    # can vary with the model
    train_iteration = model.get_train_function(index, piano_train, orchestra_train, optimizer, name='train_iteration')
    # Same for the validation
    validation_error = model.get_validation_error(index, piano_valid, orchestra_valid, name='validation_error')

    ############################################################
    ############################################################
    ############################################################
    # TRAINING
    logger_train.info("#")
    logger_train.info("# Training")
    epoch = 0
    OVERFITTING = False
    val_order = 4
    val_tab = np.zeros(val_order)
    while (not OVERFITTING or epoch!=max_iter):
        # go through the training set
        train_cost_epoch = []
        train_monitor_epoch = []
        for batch_index in xrange(n_train_batches):
            this_cost, this_monitor = train_iteration(train_index[batch_index])
            # Keep track of cost
            train_cost_epoch.append(this_cost)
            train_monitor_epoch.append(this_monitor)

        if ((epoch % 5 == 0) or (epoch < 10)):
            accuracy = []
            for batch_index in xrange(n_val_batches):
                _, _, accuracy_batch = validation_error(valid_index[batch_index])
                accuracy += [accuracy_batch]

            # Stop if validation error decreased over the last three validation
            # "FIFO" from the left
            val_tab[1:] = val_tab[0:-1]
            mean_accuracy = 100 * np.mean(accuracy)
            check_increase = np.sum(mean_accuracy >= val_tab[1:])
            if check_increase == 0:
                OVERFITTING = True
            val_tab[0] = mean_accuracy
            # Monitor learning
            logger_train.info(("Epoch : {} , Monitor : {} , Rec error : {} , Valid acc : {}"
                              .format(epoch, np.mean(train_monitor_epoch), np.mean(train_cost_epoch), mean_accuracy))
                              .encode('utf8'))

        epoch += 1

    # Return results
    best_accuracy = np.amax(val_tab)
    dico_res = model_param
    dico_res.update(optim_param)
    dico_res['accuracy'] = best_accuracy

    return dico_res


if __name__ == "__main__":
    # Check is the result folder exists
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    ############################################################
    ####  L  O  G  G  I  N  G  #################################
    ############################################################
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_file_path,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Now, we can log to the root logger, or any other logger. First the root...
    logging.info('#'*40)
    logging.info('#'*40)
    logging.info('#'*40)
    logging.info('* L * O * P *')

    # Now, define a couple of other loggers which might represent areas in your
    # application:
    logger_hopt = logging.getLogger('hyperopt')
    logger_train = logging.getLogger('train')
    logger_load = logging.getLogger('load')

    ######################################
    ###### Rebuild database
    if REBUILD_DATABASE:
        logging.info('# ** Database REBUILT **')
        PREFIX_INDEX_FOLDER = "../Data/Index/"
        index_files_dict = {}
        index_files_dict['train'] = [
            PREFIX_INDEX_FOLDER + "bouliane_train.txt",
            PREFIX_INDEX_FOLDER + "hand_picked_Spotify_train.txt",
            PREFIX_INDEX_FOLDER + "liszt_classical_archives_train.txt"
        ]
        index_files_dict['valid'] = [
            PREFIX_INDEX_FOLDER + "bouliane_valid.txt",
            PREFIX_INDEX_FOLDER + "hand_picked_Spotify_valid.txt",
            PREFIX_INDEX_FOLDER + "liszt_classical_archives_valid.txt"
        ]
        index_files_dict['test'] = [
            PREFIX_INDEX_FOLDER + "bouliane_test.txt",
            PREFIX_INDEX_FOLDER + "hand_picked_Spotify_test.txt",
            PREFIX_INDEX_FOLDER + "liszt_classical_archives_test.txt"
        ]
        build_data(index_files_dict=index_files_dict,
                   meta_info_path='temp.p',
                   quantization=quantization,
                   temporal_granularity=temporal_granularity)
    else:
        logging.info('# ** Database NOT rebuilt ** ')
    ######################################

    ######################################
    ###### HOPT function
    best = train_hopt(temporal_granularity, max_evals, log_file_path, result_file)
    logging.info(best)
    ######################################

    ######################################
    ###### Or directly call the train function for one set of HPARAMS
    # model_param = {
    #     'temporal_order': 100,
    #     'n_hidden': 150,
    #     'batch_size': 2,
    #     'gibbs_steps': 15
    # }
    # optim_param = {
    #     'lr': 0.001
    # }
    # dico_res = train(model_param,
    #                  optim_param,
    #                  temporal_granularity,
    #                  max_iter,
    #                  log_file_path)
    ######################################
