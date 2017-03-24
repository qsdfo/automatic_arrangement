#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import logging
import numpy as np
import cPickle as pkl
import time

from acidano.utils.measure import accuracy_measure_not_shared
from acidano.utils.early_stopping import up_criterion


def load_split_data(seq_length, set_identifier, data_folder):
    piano = np.load(data_folder + '/piano_' + set_identifier + '.csv')
    orch = np.load(data_folder + '/orchestra_' + set_identifier + '.csv')
    tracks_start_end = pkl.load(open(data_folder + '/tracks_start_end_' + set_identifier + '.pkl', 'rb'))
    orch_past = []
    piano_past = []
    piano_t = []
    orch_t = []
    for (start, end) in tracks_start_end.values():
        for t in range(start+seq_length,end):
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
    if script_param['unit_type'] = 'binary':
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
    if script_param['unit_type'] = 'binary':
        (model.model).compile(optimizer=script_param['optimizer'], loss='binary_crossentropy')
    elif script_param['unit_type'] = 'continuous':
        (model.model).compile(optimizer=script_param['optimizer'], loss='mean_squared_error')
    epoch = 0
    time_limit = script_param["time_limit"] * 3600 - 30*60 # walltime - 30 minutes in seconds
    OVERFITTING = False
    TIME_LIMIT = False
    val_tab = np.zeros(max(1,script_param["max_iter"]))
    start_time_train = time.time()
    while (not OVERFITTING and not TIME_LIMIT
           and epoch!=script_param["max_iter"]):

        # Fit
        history = model.fit(orch_past_train, orch_t_train, piano_past_train, piano_t_train)

        # Validation
        orch_predicted = model.validate(orch_past_valid, orch_t_valid, piano_past_valid, piano_t_valid)
        if script_param['unit_type'] == 'binary':
            accuracy = accuracy_measure_not_shared(orch_t_valid, orch_predicted)
        elif script_param['unit_type'] == 'continuous':
            accuracy = accuracy_measure_not_shared_continuous(orch_t_valid, orch_predicted)
        mean_accuracy = 100 * np.mean(accuracy)

        # Over-fitting monitor
        val_tab[epoch] = mean_accuracy
        if epoch >= script_param["min_number_iteration"]:
            OVERFITTING = up_criterion(-val_tab, epoch, script_param["number_strips"], script_param["validation_order"])

        if (time.time() - start_time_train) > time_limit:
            TIME_LIMIT = True

        logger_run.info(('Epoch : {} , Valid acc : {}'
                         .format(epoch, mean_accuracy))
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
