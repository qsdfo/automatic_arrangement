#!/usr/bin/env pyth
# -*- coding: utf8 -*-

import cPickle as pkl
import random
import logging
import glob
import os
import shutil
import time
import numpy as np
import hyperopt

from train import train
from generate_midi import generate_midi
import config
from LOP.Database.load_data_k_folds import build_folds
from LOP.Utils.normalization import get_whitening_mat, apply_pca, apply_zca
from LOP.Utils.process_data import process_data_piano, process_data_orch
from LOP.Utils.analysis_data import get_activation_ratio

# MODEL
#from LOP.Models.Future_piano.recurrent_embeddings_0 import Recurrent_embeddings_0 as Model
from LOP.Models.Real_time.LSTM_plugged_base import LSTM_plugged_base as Model

GENERATE=False
SAVE=False
DEFINED_CONFIG = True  # HYPERPARAM ?
# For reproducibility
RANDOM_SEED_FOLDS=1234 # This is useful to use always the same fold split
RANDOM_SEED=None

def main():
    # DATABASE
    DATABASE = config.data_name()
    DATABASE_PATH = config.data_root() + "/" + DATABASE
    # RESULTS
    result_folder =  config.result_root() + '/' + DATABASE + '/' + Model.name()
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    # Parameters
    parameters = config.parameters(result_folder)

    # Load the database metadata and add them to the script parameters to keep a record of the data processing pipeline
    parameters.update(pkl.load(open(DATABASE_PATH + '/metadata.pkl', 'rb')))

    ############################################################
    # Logging
    ############################################################
    # log file
    log_file_path = 'log'
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
    logging.info('#'*60)
    logging.info('#'*60)
    logging.info('#'*60)
    logging.info('* L * O * P *')
    logging.info((u'** Model : ' + Model.name()).encode('utf8'))
    for k, v in parameters.iteritems():
        logging.info((u'** ' + k + ' : ' + str(v)).encode('utf8'))
    logging.info('#'*60)
    logging.info('#'*60)

    ############################################################
    # Hyper parameter space
    ############################################################
    # Two cases :
    # 1/ Random search
    model_parameters_space = Model.get_hp_space()
    # 2/ Defined configurations
    configs = config.import_configs()
    
    # On from each database and each set
    track_paths_generation = [
        # Bouliane train
        config.database_root() + '/LOP_database_06_09_17/bouliane/0',
        # Bouliane test
        config.database_root() + '/LOP_database_06_09_17/bouliane/17',
        # Bouliane valid
        config.database_root() + '/LOP_database_06_09_17/bouliane/16',
        # Spotify train
        config.database_root() + '/LOP_database_06_09_17/hand_picked_Spotify/0',
        # Spotify test
        config.database_root() + '/LOP_database_06_09_17/hand_picked_Spotify/21',
        # Spotify valid
        config.database_root() + '/LOP_database_06_09_17/hand_picked_Spotify/20',
        # Liszt train
        config.database_root() + '/LOP_database_06_09_17/liszt_classical_archives/0',
        # Liszt test
        config.database_root() + '/LOP_database_06_09_17/liszt_classical_archives/17',
        # Liszt valid
        config.database_root() + '/LOP_database_06_09_17/liszt_classical_archives/16'
    ]

    ############################################################
    # Grid search loop
    ############################################################
    # Organisation :
    # Each config is a folder with a random ID (integer)
    # In eahc of this folder there is :
    #    - a config.pkl file with the hyper-parameter space
    #    - a result.txt file with the result
    # The result.csv file containing id;result is created from the directory, rebuilt from time to time

    if DEFINED_CONFIG:
        for config_id, model_parameters in configs.iteritems():
            config_folder = parameters['result_folder'] + '/' + config_id
            if not os.path.isdir(config_folder):
                os.mkdir(config_folder)
            else:
                # continue
                user_input = raw_input(config_folder + " folder already exists. Type y to overwrite : ")
                if user_input == 'y':
                    shutil.rmtree(config_folder)
                    os.mkdir(config_folder) 
                else:
                    raise Exception("Config not overwritten")
            config_loop(config_folder, model_parameters, parameters, DATABASE_PATH, track_paths_generation)
    else:
        # Already tested configs
        list_config_folders = glob.glob(result_folder + '/*')
        number_hp_config = max(0, parameters["max_hyperparam_configs"] - len(list_config_folders))
        for hp_config in range(number_hp_config):
            # Give a random ID and create folder
            ID_SET = False
            while not ID_SET:
                ID_config = str(random.randint(0, 2**25))
                config_folder = parameters['result_folder'] + '/' + ID_config
                if config_folder not in list_config_folders:
                    ID_SET = True
            os.mkdir(config_folder)

            # Sample model parameters from hyperparam space
            model_parameters = hyperopt.pyll.stochastic.sample(model_parameters_space)

            config_loop(config_folder, model_parameters, parameters, DATABASE_PATH, track_paths_generation)

            # Update folder list
            list_config_folders.append(config_folder)


def config_loop(config_folder, model_params, parameters, database_path, track_paths_generation):
    # New logger
    log_file_path = config_folder + '/' + 'log.txt'
    with open(log_file_path, 'wb') as f:
        f.close()
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger_config = logging.getLogger(config_folder)
    hdlr = logging.FileHandler(log_file_path)
    hdlr.setFormatter(formatter)
    logger_config.addHandler(hdlr)
    
    # Prompt model parameters
    logger_config.info('#'*60)
    logger_config.info('#### ' + config_folder)
    logger_config.info('#### Model parameters')
    logger_config.info((u'** Model : ' + Model.name()).encode('utf8'))
    for k, v in model_params.iteritems():
        logger_config.info((u'** ' + k + ' : ' + str(v)).encode('utf8'))
    
    # Get data and tvt splits (=folds)
    piano, orch, K_folds, valid_names, test_names, dimensions = get_data_and_folds(database_path, parameters, model_params, logger_config)
    pkl.dump(dimensions, open(config_folder + '/dimensions.pkl', 'wb'))

    for K_fold_ind, K_fold in enumerate(K_folds):
        config_folder_fold = config_folder + '/' + str(K_fold_ind)
        os.makedirs(config_folder_fold)
        # Write filenames of this split
        with open(os.path.join(config_folder_fold, "test_names.txt"), "wb") as f:
            for filename in test_names[K_fold_ind]:
                f.write(filename + "\n")
        with open(os.path.join(config_folder_fold, "valid_names.txt"), "wb") as f:
            for filename in valid_names[K_fold_ind]:
                f.write(filename + "\n")
        
        ## Normalize the data
        # Normalization must be computed only on training data, but performed over all the dataset
        train_indices_flat = [item for sublist in K_fold['train'] for item in sublist]
        if parameters["normalize"] is not None:
            piano_train = piano[train_indices_flat]
            epsilon = 0.0001
            if parameters["normalize"] == "standard_pca":
                mean_piano, std_piano, pca_piano, _ = get_whitening_mat(piano_train, epsilon)
                piano = apply_pca(piano, mean_piano, std_piano, pca_piano, epsilon) 
                # Save the transformations for later
                standard_pca_piano = {'mean_piano': mean_piano, 'std_piano': std_piano, 'pca_piano': pca_piano, 'epsilon': epsilon}
                pkl.dump(standard_pca_piano, open(os.path.join(config_folder_fold, "standard_pca_piano"), 'wb'))
            elif parameters["normalize"] == "standard_zca":
                mean_piano, std_piano, _, zca_piano = get_whitening_mat(piano_train, epsilon)
                piano = apply_zca(piano, mean_piano, std_piano, zca_piano, epsilon)
                # Save the transformations for later
                standard_zca_piano = {'mean_piano': mean_piano, 'std_piano': std_piano, 'zca_piano': zca_piano, 'epsilon': epsilon}
                pkl.dump(standard_zca_piano, open(os.path.join(config_folder_fold, "standard_zca_piano"), 'wb'))
            else:
                raise Exception(str(parameters["normalize"]) + " is not a possible value for normalization parameter")
        
        # Compute training data's statistics for improving learning (e.g. weighted Xent)
        activation_ratio = get_activation_ratio(orch[train_indices_flat])
        # It's okay to add this value to the parameters now because we don't need it for persistency, 
        # this is only training regularization
        model_params['activation_ratio'] = activation_ratio
        
        ########################################################
        # Persistency
        pkl.dump(model_params, open(config_folder + '/model_params.pkl', 'wb'))
        pkl.dump(Model.is_keras(), open(config_folder + '/is_keras.pkl', 'wb'))
        pkl.dump(parameters, open(config_folder + '/script_parameters.pkl', 'wb'))
        # Training
        train_wrapper(parameters, model_params, dimensions, config_folder_fold, piano, orch, K_fold, logger_config)
        # Generating
        if GENERATE:
            generate_wrapper(config_folder_fold, track_paths_generation, logger_config)
        if not SAVE:
#            shutil.rmtree(config_folder_fold + '/model')
            shutil.rmtree(config_folder_fold + '/model_Xent')
            shutil.rmtree(config_folder_fold + '/model_acc')
        ########################################################
    logger_config.info("#"*60)
    logger_config.info("#"*60)
    return


def get_data_and_folds(database_path, parameters, model_params, logger):
    logger.info((u'##### Data').encode('utf8'))
    
    # Load data and build K_folds
    time_load_0 = time.time()

    ## Load the matrices
    piano = np.load(database_path + '/piano.npy')
    orch = np.load(database_path + '/orchestra.npy')
    if parameters['duration_piano']:
        duration_piano = np.load(database_path + '/duration_piano.npy')
    else:
        duration_piano = None
    piano = process_data_piano(piano, duration_piano, parameters)
    orch = process_data_orch(orch, parameters)
    
    # ####################################################
    # ####################################################
    # # TEMP : plot random parts of the data to check alignment
    # from LOP_database.visualization.numpy_array.visualize_numpy import visualize_mat
    # T = len(piano)
    # for t in np.arange(100, T, 1000):
    #   AAA = np.concatenate((piano[t-20:t]*2, orch[t-20:t]), axis=1)
    #   visualize_mat(AAA, "debug", str(t))
    # ####################################################
    # ####################################################

    ## Load the folds
    if parameters["k_folds"] == 0:
        K_folds, valid_names, test_names = build_folds(database_path, 10, model_params["temporal_order"], parameters["batch_size"], RANDOM_SEED_FOLDS, logger_load=None)
        K_folds = [K_folds[0]]
        valid_names = [valid_names[0]]
        test_names = [test_names[0]]
    elif parameters["k_folds"] == -1:
        K_folds, valid_names, test_names = build_folds(database_path, -1, model_params["temporal_order"], parameters["batch_size"], RANDOM_SEED_FOLDS, logger_load=None)
    else:
        K_folds, valid_names, test_names = build_folds(database_path, parameters["k_folds"], model_params["temporal_order"], parameters["batch_size"], RANDOM_SEED_FOLDS, logger_load=None)
    time_load = time.time() - time_load_0

    ## Get dimensions of batches
    piano_dim = piano.shape[1]
    orch_dim = orch.shape[1]
    dimensions = {'temporal_order': model_params['temporal_order'],
                  'piano_dim': piano_dim,
                  'orch_dim': orch_dim}
    logger.info('TTT : Loading data took {} seconds'.format(time_load))
    return piano, orch, K_folds, valid_names, test_names, dimensions

def train_wrapper(parameters, model_params, dimensions, config_folder, piano, orch, K_fold, logger):
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    # TEST
    # percentage_training_set = model_params['percentage_training_set']
    # last_index = int(math.floor((percentage_training_set / float(100)) * len(train_index)))
    # train_index = train_index[:last_index]
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################

    train_index = K_fold['train']
    valid_index = K_fold['valid']

    ############################################################
    # Update train_param and model_param dicts with new information from load data
    ############################################################
    n_train_batches = len(train_index)
    n_val_batches = len(valid_index)

    logger.info((u'##### Data').encode('utf8'))
    logger.info((u'# n_train_batch :  {}'.format(n_train_batches)).encode('utf8'))
    logger.info((u'# n_val_batch :  {}'.format(n_val_batches)).encode('utf8'))

    parameters['n_train_batches'] = n_train_batches
    parameters['n_val_batches'] = n_val_batches

    ############################################################
    # Instanciate model and save folder
    ############################################################
    model = Model(model_params, dimensions)
    os.mkdir(config_folder + '/model_Xent/')
    os.mkdir(config_folder + '/model_acc/')

    ############################################################
    # Train
    ############################################################
    time_train_0 = time.time()
    best_validation_loss, best_accuracy, best_precision, best_recall, best_true_accuracy, best_f_score, best_Xent, best_epoch =\
        train(model, piano, orch, train_index, valid_index, parameters, config_folder, time_train_0, logger)
    time_train_1 = time.time()
    training_time = time_train_1-time_train_0
    logger.info('TTT : Training data took {} seconds'.format(training_time))
    logger.info((u'# Best model obtained at epoch :  {}'.format(best_epoch)).encode('utf8'))
    logger.info((u'# Loss :  {}'.format(best_validation_loss)).encode('utf8'))
    logger.info((u'# Accuracy :  {}'.format(best_accuracy)).encode('utf8'))
    logger.info((u'###################\n').encode('utf8'))

    ############################################################
    # Write result in a txt file
    ############################################################
    result_file_path = config_folder + '/result.csv'
    with open(result_file_path, 'wb') as f:
        f.write("loss;accuracy;precision;recall;true_accuracy;f_score;Xent\n" +\
                "{:d};{:.3f};{:.3f};{:.3f};{:.3f};{:.3f};{:.3f};{:.3f}".format(best_epoch, best_validation_loss, best_accuracy, best_precision, best_recall, best_true_accuracy, best_f_score, best_Xent))
    return

def generate_wrapper(config_folder, track_paths_generation, logger):
    for score_source in track_paths_generation:
            generate_midi(config_folder, score_source, number_of_version=3, duration_gen=100, rhythmic_reconstruction=False, logger_generate=logger)
    

if __name__ == '__main__':
    main()
