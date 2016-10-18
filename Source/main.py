#!/usr/bin/env python
# -*- coding: utf8 -*-

# Main script for music generation

import time
import os
import csv
import logging
import sys
import numpy as np
import cPickle as pickle
# Hyperopt
from hyperopt import fmin, tpe
# Perso
from acidano.data_processing.midi.write_midi import write_midi
from load_data import load_data
from build_data import build_data
from reconstruct_pr import reconstruct_pr

####################
# Reminder for plotting tools
# import matplotlib.pyplot as plt
# Histogram
# n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# plt.show()

####################
# Debugging compiler flags
import theano
# theano.config.optimizer = 'fast_compile'
# theano.config.mode = 'FAST_COMPILE'
# theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'off'

####################
# Select a model (path to the .py file)
# Two things define a model : it's architecture and the optimization method
# Passed in command line
if sys.argv[1] == "RBM":
    from acidano.models.lop.RBM import RBM as Model_class
elif sys.argv[1] == "cRBM":
    from acidano.models.lop.cRBM import cRBM as Model_class
elif sys.argv[1] == "FGcRBM":
    from acidano.models.lop.FGcRBM import FGcRBM as Model_class
elif sys.argv[1] == "LSTM":
    from acidano.models.lop.LSTM import LSTM as Model_class
elif sys.argv[1] == "RnnRbm":
    from acidano.models.lop.RnnRbm import RnnRbm as Model_class
else:
    print "error"
    return

if sys.argv[2] == "gradient_descent":
    from acidano.utils.optim import gradient_descent as Optimization_method
else:
    print "error"
    return

# Build data parameters :
REBUILD_DATABASE = False
# Temporal granularity and quantization
temporal_granularity = u'event_level'
binary_unit = True
quantization = 4

# Get main dir
MAIN_DIR = os.getcwd().decode('utf8') + u'/'

# Set hyperparameters (can be a grid)
result_folder = MAIN_DIR + u'../Results/' + temporal_granularity + '/' + Optimization_method.name() + '/' + Model_class.name()
result_file = result_folder + u'/hopt_results.csv'
log_file_path = result_folder + '/' + Model_class.name() + u'.log'

# Fixed hyper parameter
max_evals = 3       # number of hyper-parameter configurations evaluated
max_iter = 3      # nb max of iterations when training 1 configuration of hparams
# Config is set now, no need to modify source below for standard use

# Validation
validation_order = 5
initial_derivative_length = 10
check_derivative_length = 5

# Generation
generation_length = 50
seed_size = 10
quantization_write = quantization
############################################################################
############################################################################


def train_hopt(max_evals, csv_file_path):
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

        # Weights plotted and stored in a folder ####
        # Same for generated midi sequences #########
        weights_folder = result_folder + '/' + str(run_counter) + '/' + 'weights'
        if not os.path.isdir(weights_folder):
            os.makedirs(weights_folder)
        generated_folder = result_folder + '/' + str(run_counter) + '/generated_sequences'
        if not os.path.isdir(generated_folder):
            os.makedirs(generated_folder)
        model_folder = result_folder + '/' + str(run_counter) + '/model'
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        #############################################

        # Load data #################################
        time_load_0 = time.time()
        piano_train, orchestra_train, train_index, \
            piano_valid, orchestra_valid, valid_index, \
            piano_test, orchestra_test, test_index, generation_index \
            = load_data(model_param['temporal_order'],
                        model_param['batch_size'],
                        binary_unit=binary_unit,
                        skip_sample=1,
                        logger_load=logger_load)
        time_load_1 = time.time()
        logger_load.info('TTT : Loading data took {} seconds'.format(time_load_1-time_load_0))
        # For large datasets
        #   http://deeplearning.net/software/theano/tutorial/aliasing.html
        #   use borrow=True (avoid copying the whole matrix) ?
        #   Load as much as the GPU can handle, train then load other
        #       part of the dataset using shared_variable.set_value(new_value)
        #############################################

        # Train #####################################
        time_train_0 = time.time()
        model, dico_res = train(piano_train, orchestra_train, train_index,
                                piano_valid, orchestra_valid, valid_index,
                                model_param, optim_param, max_iter, weights_folder)
        time_train_1 = time.time()
        logger_train.info('TTT : Training data took {} seconds'.format(time_train_1-time_train_0))
        error = -dico_res['accuracy']  # Search for a min
        #############################################

        # Generate ##################################
        time_generate_0 = time.time()
        generate(model,
                 piano_test, orchestra_test, generation_index,
                 generation_length, seed_size, quantization_write,
                 generated_folder, logger_generate)
        time_generate_1 = time.time()
        logger_generate.info('TTT : Generating data took {} seconds'.format(time_generate_1-time_generate_0))
        #############################################

        # Save ######################################
        save_model_file = open(model_folder + '/model.pkl', 'wb')
        pickle.dump(model, save_model_file, protocol=pickle.HIGHEST_PROTOCOL)
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


def train(piano_train, orchestra_train, train_index,
          piano_valid, orchestra_valid, valid_index,
          model_param, optim_param, max_iter, weights_folder):
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
    logger_generate.info((u'##### Generation parameters').encode('utf8'))
    logger_generate.info((u'# generation_length : ' + str(generation_length)).encode('utf8'))
    logger_generate.info((u'# seed_size : ' + str(seed_size)).encode('utf8'))
    logger_generate.info((u'# quantization_write : ' + str(quantization_write)).encode('utf8'))

    ################################################################
    ################################################################
    ################################################################
    # DATA
    piano_dim = piano_train.get_value().shape[1]
    orchestra_dim = orchestra_train.get_value().shape[1]
    n_train_batches = len(train_index)
    n_val_batches = len(valid_index)
    logger_load.info((u'##### Data').encode('utf8'))
    logger_load.info((u'# n_train_batch :  {}'.format(n_train_batches)).encode('utf8'))
    logger_load.info((u'# n_val_batch :  {}'.format(n_val_batches)).encode('utf8'))

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
    # Compilation of the training function is encapsulated in the class since the 'givens'
    # can vary with the model
    train_iteration = model.get_train_function(piano_train, orchestra_train, optimizer, name='train_iteration')
    # Same for the validation
    validation_error = model.get_validation_error(piano_valid, orchestra_valid, name='validation_error')

    ############################################################
    ############################################################
    ############################################################
    # TRAINING
    logger_train.info("#")
    logger_train.info("# Training")
    epoch = 0
    OVERFITTING = False
    val_tab = np.zeros(max(1,max_iter))
    while (not OVERFITTING and epoch!=max_iter):
        # go through the training set
        train_cost_epoch = []
        train_monitor_epoch = []
        for batch_index in xrange(n_train_batches):
            this_cost, this_monitor = train_iteration(train_index[batch_index])
            # Keep track of cost
            train_cost_epoch.append(this_cost)
            train_monitor_epoch.append(this_monitor)
        # Validation
        accuracy = []
        for batch_index in xrange(n_val_batches):
            _, _, accuracy_batch = validation_error(valid_index[batch_index])
            accuracy += [accuracy_batch]

        # Early stopping criterion
        # Note that sum_{i=0}^{n} der = der(n) - der(0)
        # So mean over successive derivatives makes no sense
        # 1/ Get the mean derivative between 5 and 10 =
        #       \sum_{i=validation_order}^{validation_order+initial_derivative_length} E(i) - E(i-validation_order) / validation_order
        #
        # 2/ At each iteration, compare the mean derivative over the last five epochs :
        #       \sum_{i=0}^{validation_order} E(t)
        mean_accuracy = 100 * np.mean(accuracy)
        val_tab[epoch] = mean_accuracy
        if epoch == initial_derivative_length-1:
            ind = np.arange(validation_order-1, initial_derivative_length)
            increase_reference = (val_tab[ind] - val_tab[ind-validation_order+1]).sum() / (validation_order * len(ind))
        elif epoch >= initial_derivative_length:
            ind = np.arange(epoch - check_derivative_length + 1, epoch+1)
            derivative_mean = (val_tab[ind] - val_tab[ind-validation_order+1]).sum() / (validation_order * len(ind))
            # Mean derivative is less than 10% of increase reference
            if derivative_mean < 0.1 * increase_reference:
                OVERFITTING = True

        # Monitor learning
        logger_train.info(("Epoch : {} , Monitor : {} , Cost : {} , Valid acc : {}"
                          .format(epoch, np.mean(train_monitor_epoch), np.mean(train_cost_epoch), mean_accuracy))
                          .encode('utf8'))

        # Plot weights every ?? epoch
        if((epoch%20==0) or (epoch<5) or OVERFITTING):
            weights_folder_epoch = weights_folder + '/' + str(epoch)
            if not os.path.isdir(weights_folder_epoch):
                os.makedirs(weights_folder_epoch)
            model.weights_visualization(weights_folder_epoch)

        epoch += 1

    # Return results
    best_accuracy = np.amax(val_tab)
    dico_res = model_param
    dico_res.update(optim_param)
    dico_res['accuracy'] = best_accuracy

    return model, dico_res


def generate(model,
             piano, orchestra, indices,
             generation_length, seed_size, quantization_write,
             generated_folder, logger_generate):
    # Generate sequences from a trained model
    # piano, orchestra and index are data used to seed the generation
    # Note that generation length is fixed by the length of the piano input
    logger_generate.info("# Generating")

    generate_sequence = model.get_generate_function(
        piano=piano, orchestra=orchestra,
        generation_length=generation_length,
        seed_size=seed_size,
        batch_generation_size=len(indices),
        name="generate_sequence")

    # Load the mapping between pitch space and instrument
    metadata = pickle.load(open('../Data/metadata.pkl', 'rb'))
    instru_mapping = metadata['instru_mapping']

    # Given last indices, generate a batch of sequences
    (generated_sequence,) = generate_sequence(indices)
    if generated_folder is not None:
        for write_counter in xrange(generated_sequence.shape[0]):
            # Write midi
            pr_orchestra = reconstruct_pr(generated_sequence[write_counter], instru_mapping, binary_unit)
            write_path = generated_folder + '/' + str(write_counter) + '.mid'
            write_midi(pr_orchestra, quantization_write, write_path, tempo=80)

    return


if __name__ == "__main__":

    # Check is the result folder exists
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

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
    logger_generate = logging.getLogger('generate')

    ######################################
    ###### Rebuild database
    if REBUILD_DATABASE:
        logging.info('# ** Database REBUILT **')
        PREFIX_INDEX_FOLDER = MAIN_DIR + "../Data/Index/"
        index_files_dict = {}
        index_files_dict['train'] = [
            PREFIX_INDEX_FOLDER + "debug_train.txt",
            # PREFIX_INDEX_FOLDER + "bouliane_train.txt",
            # PREFIX_INDEX_FOLDER + "hand_picked_Spotify_train.txt",
            # PREFIX_INDEX_FOLDER + "liszt_classical_archives_train.txt"
        ]
        index_files_dict['valid'] = [
            PREFIX_INDEX_FOLDER + "debug_valid.txt",
            # PREFIX_INDEX_FOLDER + "bouliane_valid.txt",
            # PREFIX_INDEX_FOLDER + "hand_picked_Spotify_valid.txt",
            # PREFIX_INDEX_FOLDER + "liszt_classical_archives_valid.txt"
        ]
        index_files_dict['test'] = [
            PREFIX_INDEX_FOLDER + "debug_test.txt",
            # PREFIX_INDEX_FOLDER + "bouliane_test.txt",
            # PREFIX_INDEX_FOLDER + "hand_picked_Spotify_test.txt",
            # PREFIX_INDEX_FOLDER + "liszt_classical_archives_test.txt"
        ]
        build_data(index_files_dict=index_files_dict,
                   meta_info_path=MAIN_DIR + '../Data/temp.p',
                   quantization=quantization,
                   temporal_granularity=temporal_granularity,
                   store_folder=MAIN_DIR + '../Data')
    else:
        logging.info('# ** Database NOT rebuilt ** ')
    ######################################

    ######################################
    ###### HOPT function
    best = train_hopt(max_evals, result_file)
    logging.info(best)
    ######################################


    ######################################
    ###### Or directly call the train function for one set of HPARAMS
    # model_param = {
    #     'temporal_order': 20,
    #     'n_hidden': 150,
    #     'n_factor': 100,
    #     'batch_size': 2,
    #     'gibbs_steps': 10
    # ###### Or directly call the train function for one set of HPARAMS
    # model_param = {
    #     'temporal_order': 20,
    #     'gibbs_steps': 3,
    #     'n_hidden': 150,
    #     'n_factor': 104,
    #     'batch_size': 2,
    # }
    # optim_param = {
    #     'lr': 0.001
    # }
    # dico_res = train(model_param,
    #                  optim_param,
    #                  max_iter,
    #                  log_file_path)
    # piano_train, orchestra_train, train_index, \
    #     piano_valid, orchestra_valid, valid_index, \
    #     piano_test, orchestra_test, test_index, generation_index \
    #     = load_data(model_param['temporal_order'],
    #                 model_param['batch_size'],
    #                 generation_length=generation_length,
    #                 binary_unit=binary_unit,
    #                 skip_sample=1,
    #                 logger_load=logger_load)
    #
    # weights_folder = '../debug/weights/'
    # generated_folder = '../debug/generated/'
    # max_iter = 0
    # model, dico_res = train(piano_train, orchestra_train, train_index,
    #                         piano_valid, orchestra_valid, valid_index,
    #                         model_param, optim_param,
    #                         max_iter, weights_folder)
    #
    # generate(model=model,
    #          piano=piano_test, orchestra=orchestra_test, indices=generation_index,
    #          generation_length=generation_length, seed_size=seed_size,
    #          quantization_write=quantization_write,
    #          generated_folder=generated_folder, logger_generate=logger_generate)
    ######################################
