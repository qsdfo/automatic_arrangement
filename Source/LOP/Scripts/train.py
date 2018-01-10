#!/usr/bin/env python
# -*- coding: utf8 -*-

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras import backend as K
import numpy as np
import keras
import time
import os
from multiprocessing.pool import ThreadPool

import config
from LOP.Utils.early_stopping import up_criterion
from LOP.Utils.training_error import accuracy_low_TN_tf, bin_Xent_tf, bin_Xen_weighted_0_tf, accuracy_tf, sparsity_penalty_l1, sparsity_penalty_l2, bin_Xen_weighted_1_tf
from LOP.Utils.measure import accuracy_measure, precision_measure, recall_measure, true_accuracy_measure, f_measure, binary_cross_entropy
from LOP.Utils.build_batch import build_batch
from LOP.Utils.model_statistics import count_parameters
from LOP.Utils.Analysis.accuracy_and_binary_Xent import accuracy_and_binary_Xent
from LOP.Utils.Analysis.compare_Xent_acc_corresponding_preds import compare_Xent_acc_corresponding_preds

from asynchronous_load_mat import async_load_mat

DEBUG = False
# Note : debug sans summarize, qui pollue le tableau de variables
SUMMARIZE = False
ANALYSIS = False
# Device to use (flag direct)
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Logging device use ?
LOGGING_DEVICE = False

def validate(context, init_matrices_validation, valid_splits_batches, normalizer, parameters):
    
    sess = context['sess']
    temporal_order = context['temporal_order']
    mask_orch_ph = context['mask_orch_ph']
    inputs_ph = context['inputs_ph']
    orch_t_ph = context['orch_t_ph']
    preds_node = context['preds_node']
    loss_val_node = context['loss_val_node']
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
    Xent = []

    path_piano_matrices_valid = valid_splits_batches.keys()
    N_matrix_files = len(path_piano_matrices_valid)
    pool = ThreadPool(processes=1)
    matrices_from_thread = init_matrices_validation

    for file_ind_CURRENT in range(N_matrix_files):
        #######################################
        # Get indices and matrices to load
        #######################################
        # We train on the current matrix
        path_piano_matrix_CURRENT = path_piano_matrices_valid[file_ind_CURRENT]
        valid_index = valid_splits_batches[path_piano_matrix_CURRENT]
        # But load the next one : Useless if only one matrix, but I don't care, plenty of CPUs on lagavulin
        file_ind_NEXT = (file_ind_CURRENT+1) % N_matrix_files
        path_piano_matrix_NEXT = path_piano_matrices_valid[file_ind_NEXT]
        
        #######################################
        # Load matrix thread
        #######################################
        async_valid = pool.apply_async(async_load_mat, (normalizer, path_piano_matrix_NEXT, parameters))

        piano_transformed, orch, duration_piano, mask_orch = matrices_from_thread
    
        for batch_index in valid_index:
            # Build batch
            piano_t, piano_past, piano_future, orch_past, orch_future, orch_t, mask_orch_t = build_batch(batch_index, piano_transformed, orch, mask_orch, len(batch_index), temporal_order)
            # Input nodes
            feed_dict = {piano_t_ph: piano_t,
                        piano_past_ph: piano_past,
                        piano_future_ph: piano_future,
                        orch_past_ph: orch_past,
                        orch_future_ph: orch_future,
                        orch_t_ph: orch_t,
                        mask_orch_ph: mask_orch_t,
                        keras_learning_phase: 0}

            # Compute validation loss
            preds_batch, loss_batch = sess.run([preds_node, loss_val_node], feed_dict)
            
            val_loss += [loss_batch] * len(batch_index) # Multiply by size of batch for mean : HACKY
            Xent_batch = binary_cross_entropy(orch_t, preds_batch)
            accuracy_batch = accuracy_measure(orch_t, preds_batch)
            precision_batch = precision_measure(orch_t, preds_batch)
            recall_batch = recall_measure(orch_t, preds_batch)
            true_accuracy_batch = true_accuracy_measure(orch_t, preds_batch)
            f_score_batch = f_measure(orch_t, preds_batch)
        
            accuracy.extend(accuracy_batch)
            precision.extend(precision_batch)
            recall.extend(recall_batch)
            true_accuracy.extend(true_accuracy_batch)
            f_score.extend(f_score_batch)
            Xent.extend(Xent_batch)

        del(matrices_from_thread)
        matrices_from_thread = async_valid.get()

    return np.asarray(accuracy), np.asarray(precision), np.asarray(recall), np.asarray(val_loss), np.asarray(true_accuracy), np.asarray(f_score), np.asarray(Xent)

def train(model, train_splits_batches, valid_splits_batches, normalizer,
          parameters, config_folder, start_time_train, logger_train):

    # Time information used
    time_limit = parameters['walltime'] * 3600 - 30*60  # walltime - 30 minutes in seconds

    # Reset graph before starting training
    tf.reset_default_graph()
        
    ###### PETIT TEST VALIDATION
    # Use same validation en train set
    # piano_valid, orch_valid, valid_index = piano_train, orch_train, train_index

    if parameters['pretrained_model'] is None:
        logger_train.info((u'#### Graph'))
        start_time_building_graph = time.time()
        inputs_ph, orch_t_ph, preds_node, loss_node, loss_val_node, mask_orch_ph, train_step_node, keras_learning_phase, debug, saver = build_training_nodes(model, parameters)
        time_building_graph = time.time() - start_time_building_graph
        logger_train.info("TTT : Building the graph took {0:.2f}s".format(time_building_graph))
    else:
        logger_train.info((u'#### Graph'))
        start_time_building_graph = time.time() 
        inputs_ph, orch_t_ph, preds_node, loss_node, loss_val_node, mask_orch_ph, train_step_node, keras_learning_phase, debug, saver = load_pretrained_model(parameters['pretrained_model'])
        time_building_graph = time.time() - start_time_building_graph
        logger_train.info("TTT : Loading pretrained model took {0:.2f}s".format(time_building_graph))

    piano_t_ph, piano_past_ph, piano_future_ph, orch_past_ph, orch_future_ph = inputs_ph
    embedding_concat = debug[0]

    if SUMMARIZE:
        tf.summary.scalar('loss', loss)
    ############################################################

    ############################################################
    # Display informations about the models
    num_parameters = count_parameters(tf.get_default_graph())
    logger_train.info((u'** Num trainable parameters :  {}'.format(num_parameters)).encode('utf8'))
    with open(os.path.join(config_folder, 'num_parameters.txt'), 'wb') as ff:
        ff.write(num_parameters)

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
    val_tab_Xent = np.zeros(max(1, parameters['max_iter']))
    loss_tab = np.zeros(max(1, parameters['max_iter']))
    best_Xent = float("inf")
    best_acc = 0.
    best_loss = float("inf")
    best_epoch = None

    if parameters['memory_gpu']:
        configSession = tf.ConfigProto()
        configSession.gpu_options.per_process_gpu_memory_fraction = parameters['memory_gpu']
    else:
        configSession = None
    # with tf.Session(config=tf.ConfigProto(log_device_placement=LOGGING_DEVICE)) as sess:
    with tf.Session(config=configSession) as sess:
        
        if SUMMARIZE:
            merged_node = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(config_folder + '/summary', sess.graph)

        if model.is_keras():
            K.set_session(sess)
        
        # Initialize weights
        if parameters['pretrained_model']: 
            saver.restore(sess, parameters['pretrained_model'])
        else:
            sess.run(tf.global_variables_initializer())
            

        if DEBUG:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            
        ############################################################
        # Define the context dict used to store graphs and nodes and use them in auxiliary functions
        context = {}
        context['sess'] = sess
        context['temporal_order'] = model.temporal_order
        context['inputs_ph'] = inputs_ph
        context['orch_t_ph'] = orch_t_ph
        context['mask_orch_ph'] = mask_orch_ph
        context['preds_node'] = preds_node
        context['loss_node'] = loss_node
        context['loss_val_node'] = loss_val_node
        context['keras_learning_phase'] = keras_learning_phase
        ############################################################
        
        #######################################
        # Load first matrix
        #######################################
        path_piano_matrices_train = train_splits_batches.keys()
        N_matrix_files = len(path_piano_matrices_train)

        global_time_start = time.time()
        
        load_data_start = time.time()
        pool = ThreadPool(processes=1)
        async_train = pool.apply_async(async_load_mat, (normalizer, path_piano_matrices_train[0], parameters))
        matrices_from_thread = async_train.get()
        init_matrices_validation = matrices_from_thread
        load_data_time = time.time() - load_data_start
        logger_train.info("Load the first matrix time : " + str(load_data_time))

        if model.optimize() == False:
            # Some baseline models don't need training step optimization
            accuracy, precision, recall, val_loss, true_accuracy, f_score, Xent = validate(context, init_matrices_validation, valid_splits_batches, normalizer, parameters)
            mean_val_loss = np.mean(val_loss)
            mean_accuracy = 100 * np.mean(accuracy)
            mean_precision = 100 * np.mean(precision)
            mean_recall = 100 * np.mean(recall)
            mean_true_accuracy = 100 * np.mean(true_accuracy)
            mean_f_score = 100 * np.mean(f_score)
            mean_Xent = np.mean(Xent)
            return mean_val_loss, mean_accuracy, mean_precision, mean_recall, mean_true_accuracy, mean_f_score, mean_Xent, 0

        # Training iteration
        while (not OVERFITTING and not TIME_LIMIT
               and epoch != parameters['max_iter']):
        
            start_time_epoch = time.time()

            train_cost_epoch = []

            for file_ind_CURRENT in range(N_matrix_files):

                #######################################
                # Get indices and matrices to load
                #######################################
                # We train on the current matrix
                path_piano_matrix_CURRENT = path_piano_matrices_train[file_ind_CURRENT]
                train_index = train_splits_batches[path_piano_matrix_CURRENT]
                # But load the one next one
                file_ind_NEXT = (file_ind_CURRENT+1) % N_matrix_files
                path_piano_matrix_NEXT = path_piano_matrices_train[file_ind_NEXT]
                
                #######################################
                # Load matrix thread
                #######################################
                async_train = pool.apply_async(async_load_mat, (normalizer, path_piano_matrix_NEXT, parameters))

                piano_transformed, orch, duration_piano, mask_orch = matrices_from_thread

                #######################################
                # Train
                #######################################
                for batch_index in train_index:
                    # Build batch
                    piano_t, piano_past, piano_future, orch_past, orch_future, orch_t, mask_orch_t = build_batch(batch_index, piano_transformed, orch, mask_orch, len(batch_index), model.temporal_order)
                    
                    # Train step
                    feed_dict = {piano_t_ph: piano_t,
                                piano_past_ph: piano_past,
                                piano_future_ph: piano_future,
                                orch_past_ph: orch_past,
                                orch_future_ph: orch_future,
                                orch_t_ph: orch_t,
                                mask_orch_ph: mask_orch_t,
                                keras_learning_phase: 1}

                    if SUMMARIZE:
                        _, loss_batch, summary = sess.run([train_step_node, loss_node, merged_node], feed_dict)
                    else:
                        _, loss_batch, preds_batch, embedding_batch = sess.run([train_step_node, loss_node, preds_node, embedding_concat], feed_dict)

                    # Keep track of cost
                    train_cost_epoch.append(loss_batch)

                #######################################
                # New matrices from thread
                #######################################
                del(matrices_from_thread)
                matrices_from_thread = async_train.get()

            if SUMMARIZE:
                if (epoch<5) or (epoch%10==0):
                    # Note that summarize here only look at the variables after the last batch of the epoch
                    # If you want to look at all the batches, include it in 
                    train_writer.add_summary(summary, epoch)
     
            mean_loss = np.mean(train_cost_epoch)
            loss_tab[epoch] = mean_loss

            #######################################
            # Validate
            #######################################
            accuracy, precision, recall, val_loss, true_accuracy, f_score, Xent = validate(context, init_matrices_validation, valid_splits_batches, normalizer, parameters)
            mean_val_loss = np.mean(val_loss)
            mean_accuracy = 100 * np.mean(accuracy)
            mean_precision = 100 * np.mean(precision)
            mean_recall = 100 * np.mean(recall)
            mean_true_accuracy = 100 * np.mean(true_accuracy)
            mean_f_score = 100 * np.mean(f_score)
            mean_Xent = np.mean(Xent)
            val_tab_loss[epoch] = mean_val_loss
            val_tab_acc[epoch] = mean_accuracy
            val_tab_prec[epoch] = mean_precision
            val_tab_rec[epoch] = mean_recall
            val_tab_true_acc[epoch] = mean_true_accuracy
            val_tab_f_score[epoch] = mean_f_score
            val_tab_Xent[epoch] = mean_Xent

            end_time_epoch = time.time()
            
            #######################################
            # Overfitting ? 
            if epoch >= parameters['min_number_iteration']:
                # Based on Xentr
               # OVERFITTING = up_criterion(val_tab_Xent, epoch, parameters["number_strips"], parameters["validation_order"])
                # Based on accuracy
                # OVERFITTING = up_criterion(-val_tab_acc, epoch, parameters["number_strips"], parameters["validation_order"])
                # Use loss on validation as early stopping criterion (most logic)
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
                               Validation accuracy : {:.3f} %, precision : {:.3f} %, recall : {:.3f} % \n \
                               True_accuracy : {:.3f} %, f_score : {:.3f} %, Xent_100 : {:.3f}'
                              .format(epoch, mean_loss, mean_val_loss, mean_accuracy, mean_precision, mean_recall, mean_true_accuracy, mean_f_score, 100*mean_Xent))
                              .encode('utf8'))

            logger_train.info(('Time : {}'
                              .format(end_time_epoch - start_time_epoch))
                              .encode('utf8'))

            #######################################
            # Best model ?
            # Xent criterion
            start_time_saving = time.time()
            if mean_Xent <= best_Xent:
                logger_train.info('Save Xent')
                saver.save(sess, config_folder + "/model_Xent/model")
#                best_epoch = epoch
                best_Xent = mean_Xent
                 
                if ANALYSIS:
#                    accuracy_and_binary_Xent(context, valid_index, os.path.join(os.getcwd(), "debug/acc_Xent"), 20)
                    compare_Xent_acc_corresponding_preds(context, valid_index[:5], os.path.join(config_folder, "debug/Xent_criterion"))
       
            # Accuracy criterion
            if mean_accuracy >= best_acc:
                logger_train.info('Save Acc')
                saver.save(sess, config_folder + "/model_acc/model")
                best_acc = mean_accuracy

                if ANALYSIS:
                    compare_Xent_acc_corresponding_preds(context, valid_index[:5], os.path.join(config_folder, "debug/Acc_criterion"))

            # Loss criterion
            if mean_val_loss <= best_loss:
                logger_train.info('Save val loss')
                saver.save(sess, config_folder + "/model_loss/model")
                best_loss = mean_val_loss

            end_time_saving = time.time()
            logger_train.info('Saving time : {:.3f}'.format(end_time_saving-start_time_saving))
            #######################################

            if OVERFITTING:
                logger_train.info('OVERFITTING !!')

            if TIME_LIMIT:
                logger_train.info('TIME OUT !!')


            #######################################
            # Epoch +1
            #######################################
            epoch += 1


        # Selection criterion
        best_epoch = np.argmin(val_tab_loss[:epoch])

        # Return best accuracy
        best_accuracy = val_tab_acc[best_epoch]
        best_validation_loss = val_tab_loss[best_epoch]
        best_precision = val_tab_prec[best_epoch]
        best_recall = val_tab_rec[best_epoch]
        best_true_accuracy = val_tab_true_acc[best_epoch]
        best_f_score = val_tab_f_score[best_epoch]
        best_Xent = val_tab_Xent[best_epoch]

    return best_validation_loss, best_accuracy, best_precision, best_recall, best_true_accuracy, best_f_score, best_Xent, best_epoch


def build_training_nodes(model, parameters):
    ############################################################
    # Build nodes
    # Inputs
    piano_t_ph = tf.placeholder(tf.float32, shape=(None, model.piano_transformed_dim), name="piano_t")
    piano_past_ph = tf.placeholder(tf.float32, shape=(None, model.temporal_order-1, model.piano_transformed_dim), name="piano_past")
    piano_future_ph = tf.placeholder(tf.float32, shape=(None, model.temporal_order-1, model.piano_transformed_dim), name="piano_future")
    #
    orch_t_ph = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="orch_t")
    orch_past_ph = tf.placeholder(tf.float32, shape=(None, model.temporal_order-1, model.orch_dim), name="orch_past")
    orch_future_ph = tf.placeholder(tf.float32, shape=(None, model.temporal_order-1, model.orch_dim), name="orch_past")
    inputs_ph = (piano_t_ph, piano_past_ph, piano_future_ph, orch_past_ph, orch_future_ph)
    # Orchestral mask
    mask_orch_ph = tf.placeholder(tf.float32, shape=(None, model.orch_dim), name="mask_orch")
    # Prediction
    preds, embedding_concat = model.predict(inputs_ph)
    # TODO : remplacer cette ligne par une fonction qui prends labels et preds et qui compute la loss
    # Comme ça on pourra faire des classifier chains
    ############################################################
    
    ############################################################
    # Loss
    with tf.name_scope('loss'):
        # distance = keras.losses.binary_crossentropy(orch_t_ph, preds)
        # distance = Xent_tf(orch_t_ph, preds)
        # distance = bin_Xen_weighted_0_tf(orch_t_ph, preds, parameters['activation_ratio'])
        # distance = bin_Xen_weighted_1_tf(orch_t_ph, preds, model.tn_weight)
        # distance = accuracy_tf(orch_t_ph, preds)
        distance = accuracy_low_TN_tf(orch_t_ph, preds, weight=model.tn_weight)

        # Add sparsity constraint on the output ? Is it still loss_val or just loss :/ ???
        sparsity_coeff = model.sparsity_coeff
        sparse_loss = sparsity_penalty_l1(preds)
        # sparse_loss = sparsity_penalty_l2(preds)
        sparse_loss = tf.nn.relu(tf.reduce_sum(preds, axis=1))
        # sparse_loss = tf.keras.layers.LeakyReLU(tf.reduce_sum(preds, axis=1))

        if sparsity_coeff != 0:
            loss_val_ = distance + sparsity_coeff * sparse_loss
        else:
            loss_val_ = distance

        if parameters['mask_orch']:
            loss_masked = tf.where(mask_orch_ph==1, loss_val_, tf.zeros_like(loss_val_))
            loss_val = tf.reduce_mean(loss_masked, name="loss")
        else:
            loss_val = tf.reduce_mean(loss_val_, name="loss")
    
    # Weight decay 
    if model.weight_decay_coeff != 0:
        loss = loss_val + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * model.weight_decay_coeff
    else:
        loss = loss_val
    ############################################################
    
    ############################################################
    if model.optimize():
        # Some models don't need training
        train_step = config.optimizer().minimize(loss)
    else:
        train_step = None
        
    keras_learning_phase = K.learning_phase()
    
    ############################################################
    # Saver
    tf.add_to_collection('preds', preds)
    tf.add_to_collection('orch_t_ph', orch_t_ph)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('loss_val', loss_val)
    tf.add_to_collection('mask_orch_ph', mask_orch_ph)
    tf.add_to_collection('train_step', train_step)
    tf.add_to_collection('keras_learning_phase', keras_learning_phase)
    tf.add_to_collection('inputs_ph', piano_t_ph)
    tf.add_to_collection('inputs_ph', piano_past_ph)
    tf.add_to_collection('inputs_ph', piano_future_ph)
    tf.add_to_collection('inputs_ph', orch_past_ph)
    tf.add_to_collection('inputs_ph', orch_future_ph)
#    Debug collection
    tf.add_to_collection('debug', embedding_concat)
    debug = (embedding_concat,)
    if model.optimize():
        saver = tf.train.Saver()
    else:
        saver = None
    ############################################################
    
    return inputs_ph, orch_t_ph, preds, loss, loss_val, mask_orch_ph, train_step, keras_learning_phase, debug, saver

def load_pretrained_model(path_to_model):
    # Restore model and preds graph
    saver = tf.train.import_meta_graph(path_to_model + '.meta')

    inputs_ph = tf.get_collection('inputs_ph')
    orch_t_ph = tf.get_collection("orch_t_ph")[0]
    preds = tf.get_collection("preds")[0]
    loss = tf.get_collection("loss")[0]
    loss_val = tf.get_collection("loss_val")[0]
    mask_orch_ph = tf.get_collection("mask_orch_ph")[0]
    train_step = tf.get_collection('train_step')[0]
    keras_learning_phase = tf.get_collection("keras_learning_phase")[0]
    debug = tf.get_collection("debug")
    return inputs_ph, orch_t_ph, preds, loss, loss_val, mask_orch_ph, train_step, keras_learning_phase, debug, saver

# bias=[v.eval() for v in tf.global_variables() if v.name == "top_layer_prediction/orch_pred/bias:0"][0]
# kernel=[v.eval() for v in tf.global_variables() if v.name == "top_layer_prediction/orch_pred/kernel:0"][0]