#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras import backend as K
import numpy as np
import keras
import time
import os

from LOP.Utils.early_stopping import up_criterion
from LOP.Utils.training_error import accuracy_low_TN_tf, bin_Xent_tf, bin_Xen_weighted_0_tf, accuracy_tf
from LOP.Utils.measure import accuracy_measure, precision_measure, recall_measure, true_accuracy_measure, f_measure, binary_cross_entropy
from LOP.Utils.build_batch import build_batch
from LOP.Utils.get_statistics import count_parameters
from LOP.Utils.Analysis.accuracy_and_binary_Xent import accuracy_and_binary_Xent
from LOP.Utils.Analysis.compare_Xent_acc_corresponding_preds import compare_Xent_acc_corresponding_preds

DEBUG = False
# Note : debug sans summarize, qui pollue le tableau de variables
SUMMARIZE = False
ANALYSIS = False
# Device to use
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Logging deive use ?
LOGGING_DEVICE = False

def validate(context, valid_index):
    
    sess = context['sess']
    temporal_order = context['temporal_order']
    piano = context['piano']
    orch = context['orch']
    inputs_ph = context['inputs_ph']
    orch_t_ph = context['orch_t_ph']
    preds = context['preds']
    loss = context['loss']
    keras_learning_phase = context['keras_learning_phase']
    
    #######################################
    # Validate
    #######################################
    piano_t_ph, piano_past_ph, piano_future_ph, orch_past_ph, orch_future_ph = inputs_ph
    accuracy = []
    precision = []
    recall = []
    val_loss = []
    true_accuracy = []
    f_score = []
    for batch_index in valid_index:
        # Build batch
        piano_t, piano_past, piano_future, orch_past, orch_future, orch_t = build_batch(batch_index, piano, orch, len(batch_index), temporal_order)
        # Input nodes
        feed_dict = {piano_t_ph: piano_t,
                    piano_past_ph: piano_past,
                    piano_future_ph: piano_future,
                    orch_past_ph: orch_past,
                    orch_future_ph: orch_future,
                    orch_t_ph: orch_t,
                    keras_learning_phase: 0}
        # Compute validation loss
        preds_batch, loss_batch = sess.run([preds, loss], feed_dict)
        val_loss += [loss_batch] * len(batch_index) # Multiply by size of batch for mean : HACKY
#        Xent_batch = binary_cross_entropy(orch_t, preds_batch)
        accuracy_batch = accuracy_measure(orch_t, preds_batch)
        precision_batch = precision_measure(orch_t, preds_batch)
        recall_batch = recall_measure(orch_t, preds_batch)
        true_accuracy_batch = true_accuracy_measure(orch_t, preds_batch)
        f_score_batch = f_measure(orch_t, preds_batch)
        
#        val_loss.extend(Xent_batch)
        accuracy.extend(accuracy_batch)
        precision.extend(precision_batch)
        recall.extend(recall_batch)
        true_accuracy.extend(true_accuracy_batch)
        f_score.extend(f_score_batch)
                
    return np.asarray(accuracy), np.asarray(precision), np.asarray(recall), np.asarray(val_loss), np.asarray(true_accuracy), np.asarray(f_score)

def train(model, piano, orch, train_index, valid_index,
          parameters, config_folder, start_time_train, logger_train):
   
    # Time information used
    time_limit = parameters['walltime'] * 3600 - 30*60  # walltime - 30 minutes in seconds

    # Reset graph before starting training
    tf.reset_default_graph()
        
    ###### PETIT TEST VALIDATION
    # Use same validation en train set
    # piano_valid, orch_valid, valid_index = piano_train, orch_train, train_index

    ############################################################
    # Compute train step
    # Inputs
    logger_train.info((u'#### Graph'))
    start_time_building_graph = time.time()
    #
    piano_t_ph = tf.placeholder(tf.float32, shape=(None, model.piano_dim), name="piano_t")
    piano_past_ph = tf.placeholder(tf.float32, shape=(None, model.temporal_order-1, model.piano_dim), name="piano_past")
    piano_future_ph = tf.placeholder(tf.float32, shape=(None, model.temporal_order-1, model.piano_dim), name="piano_future")
    #
    orch_t_ph = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="orch_t")
    orch_past_ph = tf.placeholder(tf.float32, shape=(None, model.temporal_order-1, model.orch_dim), name="orch_past")
    orch_future_ph = tf.placeholder(tf.float32, shape=(None, model.temporal_order-1, model.orch_dim), name="orch_past")
    inputs_ph = (piano_t_ph, piano_past_ph, piano_future_ph, orch_past_ph, orch_future_ph)
    # Prediction
    preds = model.predict(inputs_ph)
    # TODO : remplacer cette ligne par une fonction qui prends labels et preds et qui compute la loss
    # Comme ça on pourra faire des classifier chains
    
    # Loss
    loss = tf.reduce_mean(keras.losses.binary_crossentropy(orch_t_ph, preds), name="loss")
#    loss = tf.reduce_mean(Xent_tf(orch_t_ph, preds), name="loss") 
#    loss = tf.reduce_mean(bin_Xen_weighted_0_tf(orch_t_ph, preds, parameters['activation_ratio']), name="loss")
#    loss = tf.reduce_mean(accuracy_tf(orch_t_ph, preds), name="loss")
#    loss = tf.reduce_mean(-accuracy_low_TN_tf(orch_t_ph, preds, weight=1./500), name="loss")
    
    # train_step = tf.train.AdamOptimizer(0.5).minimize(loss)
    if model.optimize():
        # Some models don't need training
        train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    keras_learning_phase = K.learning_phase()
    time_building_graph = time.time() - start_time_building_graph
    logger_train.info("TTT : Building the graph took {0:.2f}s".format(time_building_graph))
    ############################################################

    ############################################################
    # Saver
    tf.add_to_collection('preds', preds)
    tf.add_to_collection('keras_learning_phase', keras_learning_phase)
    tf.add_to_collection('inputs_ph', piano_t_ph)
    tf.add_to_collection('inputs_ph', piano_past_ph)
    tf.add_to_collection('inputs_ph', piano_future_ph)
    tf.add_to_collection('inputs_ph', orch_past_ph)
    tf.add_to_collection('inputs_ph', orch_future_ph)
    if model.optimize():
        saver = tf.train.Saver()
    ############################################################

    ############################################################
    # Display informations about the models
    num_parameters = count_parameters(tf.get_default_graph())
    logger_train.info((u'** Num trainable parameters :  {}'.format(num_parameters)).encode('utf8'))

    ############################################################
    # Training
    logger_train.info("#" * 60)
    logger_train.info("#### Training")
    epoch = 0
    OVERFITTING = False
    TIME_LIMIT = False
    val_tab_acc = np.zeros(max(1, parameters['max_iter']))
    val_tab_prec = np.zeros(max(1, parameters['max_iter']))
    val_tab_rec = np.zeros(max(1, parameters['max_iter']))
    val_tab_loss = np.zeros(max(1, parameters['max_iter']))
    val_tab_true_acc = np.zeros(max(1, parameters['max_iter']))
    val_tab_f_score = np.zeros(max(1, parameters['max_iter']))
    loss_tab = np.zeros(max(1, parameters['max_iter']))
    best_val_loss = float("inf")
    best_acc = 0.
    best_epoch = None

    # with tf.Session(config=tf.ConfigProto(log_device_placement=LOGGING_DEVICE)) as sess:        
    with tf.Session() as sess:
        
        if SUMMARIZE: 
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(config_folder + '/summary', sess.graph)

        if model.is_keras():
            K.set_session(sess)

        # Initialize weights
        sess.run(tf.global_variables_initializer())

        if DEBUG:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            
        ############################################################
        # Define the context dict used to store graphs and nodes and use them in auxiliary functions
        context = {}
        context['sess'] = sess
        context['temporal_order'] = model.temporal_order
        context['piano'] = piano
        context['orch'] = orch
        context['inputs_ph'] = inputs_ph
        context['orch_t_ph'] = orch_t_ph
        context['preds'] = preds
        context['loss'] = loss
        context['keras_learning_phase'] = keras_learning_phase
        ############################################################
        
        if model.optimize() == False:
            # Some baseline models don't need training step optimization
            accuracy, precision, recall, val_loss, true_accuracy, f_score = validate(context, valid_index)
            mean_val_loss = np.mean(val_loss)
            mean_accuracy = 100 * np.mean(accuracy)
            mean_precision = 100 * np.mean(precision)
            mean_recall = 100 * np.mean(recall)
            mean_true_accuracy = 100 * np.mean(true_accuracy)
            mean_f_score = 100 * np.mean(f_score)
            return mean_val_loss, mean_accuracy, mean_precision, mean_recall, mean_true_accuracy, mean_f_score, 0
        
        # Training iteration
        while (not OVERFITTING and not TIME_LIMIT
               and epoch != parameters['max_iter']):

            start_time_epoch = time.time()

            #######################################
            # Train
            #######################################
            train_cost_epoch = []           
            for batch_index in train_index:
                # Build batch
                piano_t, piano_past, piano_future, orch_past, orch_future, orch_t = build_batch(batch_index, piano, orch, len(batch_index), model.temporal_order)
                
                # Train step
                feed_dict = {piano_t_ph: piano_t,
                            piano_past_ph: piano_past,
                            piano_future_ph: piano_future,
                            orch_past_ph: orch_past,
                            orch_future_ph: orch_future,
                            orch_t_ph: orch_t,
                            keras_learning_phase: 1}

                if SUMMARIZE:
                    _, loss_batch, summary = sess.run([train_step, loss, merged], feed_dict)
                else:
                    _, loss_batch = sess.run([train_step, loss], feed_dict)

                # Keep track of cost
                train_cost_epoch.append(loss_batch)

            if SUMMARIZE:
                if (epoch<5) or (epoch%10==0):
                    train_writer.add_summary(summary, epoch)
 
            mean_loss = np.mean(train_cost_epoch)
            loss_tab[epoch] = mean_loss


            #######################################
            # Validate
            #######################################
            accuracy, precision, recall, val_loss, true_accuracy, f_score = validate(context, valid_index)
            mean_val_loss = np.mean(val_loss)
            mean_accuracy = 100 * np.mean(accuracy)
            mean_precision = 100 * np.mean(precision)
            mean_recall = 100 * np.mean(recall)
            mean_true_accuracy = 100 * np.mean(true_accuracy)
            mean_f_score = 100 * np.mean(f_score)
            val_tab_loss[epoch] = mean_val_loss
            val_tab_acc[epoch] = mean_accuracy
            val_tab_prec[epoch] = mean_precision
            val_tab_rec[epoch] = mean_recall
            val_tab_true_acc[epoch] = mean_true_accuracy
            val_tab_f_score[epoch] = mean_f_score

            end_time_epoch = time.time()
            
            #######################################
            # Overfitting ? Based on Xentr
            if epoch >= parameters['min_number_iteration']:
                OVERFITTING = up_criterion(val_tab_loss, epoch, parameters["number_strips"], parameters["validation_order"])
            #######################################

            #######################################
            # Monitor time (guillimin walltime)
            if (time.time() - start_time_train) > time_limit:
                TIME_LIMIT = True
            #######################################

            #######################################
            # Log training
            #######################################
            logger_train.info("############################################################")
            logger_train.info(('Epoch : {} , Training loss : {} , Validation loss : {} \n \
                               Validation accuracy : {} %, precision : {} %, recall : {} % \n \
                               True_accuracy : {} %, f_score : {} %'
                              .format(epoch, mean_loss, mean_val_loss, mean_accuracy, mean_precision, mean_recall, mean_true_accuracy, mean_f_score))
                              .encode('utf8'))

            logger_train.info(('Time : {}'
                              .format(end_time_epoch - start_time_epoch))
                              .encode('utf8'))

            #######################################
            # Best model ?
            # Xent criterion
            if mean_val_loss <= best_val_loss:
                save_time_start = time.time()
                saver.save(sess, config_folder + "/model_Xent/model")
                best_epoch = epoch
                save_time = time.time() - save_time_start
                logger_train.info(('Save time : {}'.format(save_time)).encode('utf8'))
                best_val_loss = mean_val_loss
                
                # Analysis
                if ANALYSIS:
#                    accuracy_and_binary_Xent(context, valid_index, os.path.join(os.getcwd(), "debug/acc_Xent"), 20)
                    compare_Xent_acc_corresponding_preds(context, valid_index[:5], os.path.join(config_folder, "debug/Xent_criterion"))
            # Accuracy criterion
            if mean_accuracy >= best_acc:
                saver.save(sess, config_folder + "/model_acc/model")
                best_acc = mean_accuracy
                if ANALYSIS:
                    compare_Xent_acc_corresponding_preds(context, valid_index[:5], os.path.join(config_folder, "debug/Acc_criterion"))
            #######################################

            if OVERFITTING:
                logger_train.info('OVERFITTING !!')

            if TIME_LIMIT:
                logger_train.info('TIME OUT !!')

            #######################################
            # Epoch +1
            #######################################
            epoch += 1

        # Return best accuracy
        best_accuracy = val_tab_acc[best_epoch]
        best_validation_loss = val_tab_loss[best_epoch]
        best_precision = val_tab_prec[best_epoch]
        best_recall = val_tab_rec[best_epoch]
        best_true_accuracy = val_tab_true_acc[best_epoch]
        best_f_score = val_tab_f_score[best_epoch]

    return best_validation_loss, best_accuracy, best_precision, best_recall, best_true_accuracy, best_f_score, best_epoch