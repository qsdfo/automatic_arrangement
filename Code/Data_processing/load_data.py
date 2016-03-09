#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import cPickle
import theano

from pianoroll_reduction import remove_unused_pitch
from minibatch_builder import k_fold_cross_validation
from event_level import get_event_ind

# from matplotlib import pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages


def get_data(data_path, log_file_path, temporal_granularity, temporal_order, shared_bool=True):
    """
    Load data from pickle (.p) file into a matrix. Return valid indexes for sequential learning

    data_path:      relative path to pickelized data
    temporal_order: temporal order of the model for which data are loaded

    data :          shared variable containing the pianoroll representation of all the file concatenated in
                    a single huge pr
    valide_indexes: list of valid training point given the temporal order

    """

    # Open log file
    log_file = open(log_file_path, "ab")
    log_file.write("### LOADING DATA ###\n")
    log_file.write("## Unpickle data... ")

    data_all = cPickle.load(open(data_path, "rb"))
    log_file.write("Done !\n")

    quantization = data_all['quantization']
    log_file.write("## Quantization : %d\n" % quantization)

    # Build the pianoroll and valid indexes
    log_file.write("## Reading scores :\n")
    scores = data_all['scores']
    instru_mapping = data_all['instru_mapping']
    orchestra_dimension = data_all['orchestra_dimension']
    piano_dimension = instru_mapping['Piano'][1] - instru_mapping['Piano'][0]   # Should be 128
    # Time increment variables
    last_stop = 0
    start_time = 0
    end_time = 0
    # Keep trace of the valid indices
    valid_index = []
    # Write flags
    time_updated = False
    orch_init = False
    for info in scores.itervalues():
        # Concatenate pianoroll along time dimension
        # Construct an auxiliary pianoroll
        for instru, pianoroll_instru in info['pianoroll'].iteritems():
            if not time_updated:
                # set start and end time
                start_time = end_time
                end_time = start_time + np.shape(pianoroll_instru)[0]
                time_updated = True
            if instru == 'Piano':
                # concatenate empty pianoroll
                aux_piano = pianoroll_instru
            else:
                # if 'orch' not in locals():
                #     aux_orch = np.zeros([end_time, orchestra_dimension])
                #     orch_init = True
                if not orch_init:
                    # concatenate empty pianoroll
                    aux_orch = np.zeros([end_time - start_time, orchestra_dimension])
                    # orch = np.concatenate((orch, empty_orch), axis=0)
                    orch_init = True
                instru_ind_start = instru_mapping[instru][0]
                instru_ind_end = instru_mapping[instru][1]
                aux_orch[0:end_time - start_time, instru_ind_start:instru_ind_end] = pianoroll_instru
        if temporal_granularity == u'full_event_level':
            # in a full event_level granularity, we immediately split the database
            event_ind = get_event_ind(aux_orch)
            aux_orch = aux_orch[event_ind, :]
            aux_piano = aux_piano[event_ind, :]
        if 'piano' not in locals():
            piano = aux_piano
        else:
            piano = np.concatenate((piano, aux_piano), axis=0)
        if 'orch' not in locals():
            orch = aux_orch
        else:
            orch = np.concatenate((orch, aux_orch), axis=0)
        # Valid indexes
        N_pr = orch.shape[0]
        valid_index.extend(range(last_stop + temporal_order, N_pr))
        last_stop = N_pr
        # Set flag
        orch_init = False
        time_updated = False

        log_file.write("    # Score '%s' : " % info['filename'])
        log_file.write(" written.\n     Time indices : {} -> {}\n".format(start_time, end_time))

    log_file.write("\nDimension of the orchestra : time = {} pitch = {}".format(np.shape(orch)[0], np.shape(orch)[1]))
    log_file.write("\nDimension of the piano : time = {} pitch = {}".format(np.shape(piano)[0], np.shape(piano)[1]))

    # Remove unused pitches
    log_file.write("\n## Remove unused pitches")
    orch_clean, orch_mapping = remove_unused_pitch(orch, instru_mapping)
    piano_clean, piano_mapping = remove_unused_pitch(piano, {'Piano': (0, 128)})
    log_file.write("\nDimension of reduced orchestra : time = {} pitch = {}".format(np.shape(orch_clean)[0], np.shape(orch_clean)[1]))
    log_file.write("\nDimension of reduced piano : time = {} pitch = {}".format(np.shape(piano_clean)[0], np.shape(piano_clean)[1]))
    log_file.write("\n")

    # Event level indices
    if temporal_granularity == u'event_level':
        event_ind = get_event_ind(orch_clean)
        valid_index = set(event_ind).intersection(valid_index)

    if shared_bool:
        # Instanciate shared variables
        orch_shared = theano.shared(np.asarray(orch_clean, dtype=theano.config.floatX))
        piano_shared = theano.shared(np.asarray(piano_clean, dtype=theano.config.floatX))

    log_file.close()
    return orch_shared, orch_mapping, piano_shared, piano_mapping, np.array(valid_index), quantization


def load_data(data_path, log_file_path, temporal_granularity, temporal_order, shared_bool, minibatch_size, split=(0.7, 0.1, 0.2)):
    orch, orch_mapping, piano, piano_mapping, valid_index, quantization = get_data(data_path, log_file_path, temporal_granularity, temporal_order, shared_bool)
    import pdb; pdb.set_trace
    train_index, validate_index, test_index = k_fold_cross_validation(log_file_path, valid_index, minibatch_size, split)
    # train_index, validate_index, test_index = tvt_minibatch(log_file_path, valid_index, minibatch_size, shuffle, split)
    return orch, orch_mapping, piano, piano_mapping, train_index, validate_index, test_index


if __name__ == '__main__':
    # train_batch_ind, validate_batch_ind, test_batch_ind = tvt_minibatch('test', np.arange(20), 3, True)
    # print(train_batch_ind)
    # print(validate_batch_ind)
    # print(test_batch_ind)

    # aaa = np.transpose(np.array([[0.0, 0.0, 0.4, 0.4, 0.7],
    #                              [0.0, 0.0, 0.2, 0.2, 0.7],
    #                              [0.0, 0.0, 0.4, 0.4, 0.7]]))
    # import pdb; pdb.set_trace()
    # bbb=get_event_ind(aaa)

    orch, orch_mapping, piano, piano_mapping, train_batch_ind, validate_batch_ind, test_batch_ind = load_data(data_path='../../Data/data.p', log_file_path='log_test.txt', temporal_granularity='event_level', temporal_order=8, shared_bool=True, minibatch_size=100, shuffle=True)
