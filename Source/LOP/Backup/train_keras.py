#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import logging
import numpy as np
import cPickle as pkl
import time
import os

from acidano.utils.measure import accuracy_measure_not_shared, accuracy_measure_not_shared_continuous
from acidano.utils.early_stopping import up_criterion
from acidano.models.lop_keras.plot_weights import save_weights


def load_split_data(seq_length, set_identifier, data_folder):
    piano = np.load(data_folder + '/piano_' + set_identifier + '.csv')
    orch = np.load(data_folder + '/orchestra_' + set_identifier + '.csv')
    tracks_start_end = pkl.load(open(data_folder + '/tracks_start_end_' + set_identifier + '.pkl', 'rb'))
    orch_past = []
    piano_past = []
    piano_t = []
    orch_t = []
    for (start, end) in tracks_start_end.values():
        for t in range(start+seq_length, end):
            # We predict t with t-seq_length to t-1
            orch_past.append(orch[t-seq_length:t])
            piano_past.append(piano[t-seq_length:t])
            piano_t.append(piano[t])
            orch_t.append(orch[t])
    return np.asarray(orch_past), np.asarray(orch_t), np.asarray(piano_past), np.asarray(piano_t)


def run_wrapper(params, config_folder, start_time_train):
    ############################################################
    # Load h params
    ############################################################
    script_param = params['script']
    model_param = params['model']
    if script_param['unit_type'] == 'binary':
        model_param['binary'] = True
    else:
        model_param['binary'] = False

    ############################################################
    # Log file
    ############################################################
    log_file_path = config_folder + '/log.txt'
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger_run = logging.getLogger('run')
    hdlr = logging.FileHandler(log_file_path)
    hdlr.setFormatter(formatter)
    logger_run.addHandler(hdlr)

    ############################################################
    # Load data
    ############################################################
    logger_run.info('########################################################')
    logger_run.info('Loading data')
    # Build matrices
    orch_past_train, orch_t_train, piano_past_train, piano_t_train = load_split_data(model_param['temporal_order'], 'train', script_param["data_folder"])
    orch_past_valid, orch_t_valid, piano_past_valid, piano_t_valid = load_split_data(model_param['temporal_order'], 'valid', script_param["data_folder"])
    # Get dimensions
    dimensions = {}
    dimensions["piano_dim"] = piano_t_train.shape[-1]
    dimensions["orch_dim"] = orch_t_train.shape[-1]
    for k, v in dimensions.iteritems():
        logger_run.info(k + " : " + str(v))

    ############################################################
    # Instanciate the model
    ############################################################
    logger_run.info('########################################################')
    logger_run.info('Loading model')
    if script_param["model_name"] == 'Lstm':
        from acidano.models.lop_keras.binary.lstm import Lstm as Model_class
    elif script_param["model_name"] == 'repeat':
        from acidano.models.lop_keras.binary.repeat import Repeat as Model_class

    model = Model_class(model_param, dimensions)
    model.build_model()
    for k, v in model_param.iteritems():
        logger_run.info(k + " : " + str(v))

    ############################################################
    # Train
    ############################################################
    logger_run.info('########################################################')
    logger_run.info('Training...')
    # Train
    if model.model:
        # Some model (repeat for instance) don't have a model.model variable
        if script_param['unit_type'] == 'binary':
            (model.model).compile(optimizer=script_param['optimizer'], loss='binary_crossentropy')
        elif script_param['unit_type'] == 'continuous':
            (model.model).compile(optimizer=script_param['optimizer'], loss='mean_squared_error')
    epoch = 0
    time_limit = script_param["time_limit"] * 3600 - 30*60  # walltime - 30 minutes in seconds
    OVERFITTING = False
    TIME_LIMIT = False
    val_tab = np.zeros(max(1, script_param["max_iter"]))
    start_time_train = time.time()
    while (not OVERFITTING and not TIME_LIMIT
           and (epoch != script_param["max_iter"])):

        # Fit
        history = model.fit(orch_past_train, orch_t_train, piano_past_train, piano_t_train)
        if history:
            loss = history.history['loss'][0]
        else:
            # Some model (repeat for instance don't have a training step, hence no history dict)
            loss = 0

        # Validation
        orch_predicted = model.validate(orch_past_valid, orch_t_valid, piano_past_valid, piano_t_valid)
        if script_param['unit_type'] == 'binary':
            accuracy = accuracy_measure_not_shared(orch_t_valid, orch_predicted)
        elif script_param['unit_type'] == 'continuous':
            accuracy = accuracy_measure_not_shared_continuous(orch_t_valid, orch_predicted)
        mean_accuracy = 100 * np.mean(accuracy)

        ####################################################################################
        ####################################################################################
        ####################################################################################
        ####################################################################################
        if script_param['DEBUG']:
            # Weights
            plot_folder = config_folder + '/DEBUG/' + str(epoch) + '/weights'
            if not os.path.isdir(plot_folder):
                os.makedirs(plot_folder)
            save_weights(model.model, plot_folder)

            # Variables
            from acidano.visualization.numpy_array.visualize_numpy import visualize_mat
            for ind in range(0, accuracy.shape[0], 1000):
                pr_viz = np.zeros((3, orch_t_valid.shape[1]))
                # Threshold prediction
                orch_pred_ind = orch_predicted[ind]
                # Less than 1%
                thresh_pred = np.where(orch_pred_ind > 0.01, orch_pred_ind, 0)
                pr_viz[0] = thresh_pred
                pr_viz[1] = orch_t_valid[ind]
                pr_viz[2] = orch_past_valid[ind, -1, :]
                path_accuracy = config_folder + '/DEBUG/' + str(epoch) + '/validation'
                if not os.path.isdir(path_accuracy):
                    os.makedirs(path_accuracy)
                visualize_mat(np.transpose(pr_viz), path_accuracy, str(ind) + '_score_' + str(accuracy[ind]))
        ####################################################################################
        ####################################################################################
        ####################################################################################
        ####################################################################################

        # Over-fitting monitor
        val_tab[epoch] = mean_accuracy
        if epoch >= script_param["min_number_iteration"]:
            OVERFITTING = up_criterion(-val_tab, epoch, script_param["number_strips"], script_param["validation_order"])

        if (time.time() - start_time_train) > time_limit:
            TIME_LIMIT = True

        logger_run.info(('Epoch : {}, Loss : {}, Valid acc : {}'
                        .format(epoch, loss, mean_accuracy))
                        .encode('utf8'))

        if OVERFITTING:
            logger_run.info('OVERFITTING !!')
        if TIME_LIMIT:
            logger_run.info('TIME OUT !!')
        epoch += 1

    ############################################################
    # Save
    ############################################################
    # model
    if model.model:
        (model.model).save(config_folder + '/model.h5')
    # result
    best_epoch = np.argmax(val_tab)
    best_accuracy = val_tab[best_epoch]
    result_file_path = config_folder + '/result.csv'
    with open(result_file_path, 'wb') as f:
        f.write("accuracy;" + str(best_accuracy))

    # Remove handler
    logger_run.removeHandler(hdlr)


if __name__ == '__main__':
    start_time_train = time.time()
    config_folder = sys.argv[1]
    params = pkl.load(open(config_folder + '/config.pkl', "rb"))
    run_wrapper(params, config_folder, start_time_train)
