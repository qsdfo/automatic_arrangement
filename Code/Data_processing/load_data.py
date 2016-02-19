#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import cPickle
import theano


def get_data(data_path, log_file_path, temporal_order):
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
    orch_concatenated = False
    for info in scores.itervalues():
        # Concatenate pianoroll along time dimension
        for instru, pianoroll_instru in info['pianoroll'].iteritems():
            if not time_updated:
                # set start and end time
                start_time = end_time
                end_time = start_time + np.shape(pianoroll_instru)[0]
                time_updated = True
            if instru == 'Piano':
                if 'piano' not in locals():
                    # Initialize piano pr
                    piano = np.zeros([end_time, piano_dimension])
                else:
                    # concatenate empty pianoroll
                    empty_piano = np.zeros([end_time - start_time, piano_dimension])
                    piano = np.concatenate((piano, empty_piano), axis=0)
                piano[start_time:end_time, 0:piano_dimension] = pianoroll_instru
            else:
                if 'orch' not in locals():
                    orch = np.zeros([end_time, orchestra_dimension])
                    orch_concatenated = True
                if not orch_concatenated:
                    # concatenate empty pianoroll
                    empty_orch = np.zeros([end_time - start_time, orchestra_dimension])
                    orch = np.concatenate((orch, empty_orch), axis=0)
                    orch_concatenated = True
                instru_ind_start = instru_mapping[instru][0]
                instru_ind_end = instru_mapping[instru][1]
                orch[start_time:end_time, instru_ind_start:instru_ind_end] = pianoroll_instru
        # Valid indexes
        N_pr = orch.shape[0]
        valid_index.extend(range(last_stop + temporal_order, N_pr))
        last_stop = N_pr
        # Set flag
        orch_concatenated = False
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

    ################################
    ################################
    ################################
    # DEBUG
    # notes = mat_to_csv(piano, quantization)
    # with open('piano.csv', 'wb') as f_handle:
    #     writer = csv.writer(f_handle, delimiter=',')
    #     writer.writerows(notes)
    # notes = mat_to_csv(piano, quantization)
    # with open('piano.csv', 'wb') as f_handle:
    #     writer = csv.writer(f_handle, delimiter=',')
    #     writer.writerows(notes)
    # np.savetxt("DEBUG/orch.csv", orch, delimiter=",", fmt='%1.4f')
    # np.savetxt("DEBUG/piano.csv", piano, delimiter=",", fmt='%1.4f')
    # np.savetxt("DEBUG/indices.csv", valid_index, delimiter=",", fmt='%1.4f')
    # import pdb; pdb.set_trace()
    ################################
    ################################
    ################################
    orch_shared = theano.shared(np.asarray(orch_clean, dtype=theano.config.floatX))
    piano_shared = theano.shared(np.asarray(piano_clean, dtype=theano.config.floatX))

    log_file.close()
    return orch_shared, orch_mapping, piano_shared, piano_mapping, np.array(valid_index), quantization, orch, piano


def get_minibatches_idx(log_file_path, idx_list, minibatch_size, shuffle=False, split=(0.7, 0.1, 0.2)):
    """
    Used to shuffle the dataset at each iteration.

    idx_list:   python list
                contains the valid indexes, i.e. indexes which more than temporal_order after the beginning
                of a track
    shuffle:    we might not want to shuffle the data, for monitoring purposes (?)
    split:      split proportion of the whole dataset between train, validate and test datasets
    """

    if shuffle:
        np.random.shuffle(idx_list)

    n = idx_list.shape[0]

    minibatch_start = 0

    # TRAIN
    minibatches_train = []
    last_train = int(n * split[0])
    n_batch_train = int((n * split[0]) // minibatch_size)
    for i in range(n_batch_train):
        minibatches_train.append(idx_list[minibatch_start:
                                 minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != last_train):
        # Make a minibatch out of what is left
        minibatches_train.append(idx_list[minibatch_start:last_train])
        n_batch_train = n_batch_train + 1

    # VALIDATE
    minibatches_validate = []
    minibatch_start = last_train
    last_validate = int(last_train + n * split[1])
    n_batch_validate = int((n * split[1]) // minibatch_size)
    for i in range(n_batch_validate):
        minibatches_validate.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != last_validate):
        # Make a minibatch out of what is left
        minibatches_validate.append(idx_list[minibatch_start:last_validate])
        n_batch_validate = n_batch_validate + 1

    # TEST
    minibatches_test = []
    minibatch_start = last_validate
    last_test = int(last_validate + n * split[2])
    n_batch_test = int((n * split[2]) // minibatch_size)
    for i in range(n_batch_test):
        minibatches_test.append(idx_list[minibatch_start:
                                minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != last_test):
        # Make a minibatch out of what is left
        minibatches_test.append(idx_list[minibatch_start:last_test])
        n_batch_test = n_batch_test + 1

    return np.array(minibatches_train), np.array(minibatches_validate), np.array(minibatches_test)


def remove_unused_pitch(pr, mapping):
    mapping_reduced = {}
    start_index = 0
    # First build the mapping
    for instru, pitch_range in mapping.iteritems():
        # Extract the pianoroll of this instrument
        start_index_pr = pitch_range[0]
        end_index_pr = pitch_range[1]
        pr_instru = pr[:, start_index_pr:end_index_pr]
        # Get lowest/highest used pitch
        existing_pitches = np.nonzero(pr_instru)[1]
        if np.shape(existing_pitches)[0] > 0:
            start_pitch = np.amin(existing_pitches)
            end_pitch = np.amax(existing_pitches)
            end_index = start_index + end_pitch - start_pitch
            mapping_reduced[instru] = {'start_pitch': start_pitch,
                                       'end_pitch': end_pitch,
                                       'start_index': start_index,
                                       'end_index': end_index}
            if 'pr_reduced' in locals():
                pr_reduced = np.concatenate((pr_reduced, pr_instru[:, start_pitch:end_pitch]), axis=1)
            else:
                pr_reduced = pr_instru[:, start_pitch:end_pitch]
            start_index = end_index
    return pr_reduced, mapping_reduced


def load_data(data_path, log_file_path, temporal_order, minibatch_size, shuffle=False, split=(0.7, 0.1, 0.2)):
    orch, orch_mapping, piano, piano_mapping, valid_index, quantization, orch_2, piano_2 = get_data(data_path, log_file_path, temporal_order)
    train_index, validate_index, test_index = get_minibatches_idx(log_file_path, valid_index, minibatch_size, shuffle, split)
    # #########################
    #  DEBUG
    # batch_index = 0
    # n_dim = orch_2.shape[1]
    # hist_idx = np.array([train_index[batch_index] - n for n in xrange(1, temporal_order + 1)]).T
    # A = piano_2[train_index[batch_index]]
    # B = orch_2[hist_idx.ravel()]
    # orch_reshape = B.reshape((minibatch_size, temporal_order * n_dim))
    # past = np.concatenate((A, orch_reshape), axis=1)
    # #########################
    return orch, orch_mapping, piano, piano_mapping, train_index, validate_index, test_index


if __name__ == '__main__':
    # train_batch_ind, validate_batch_ind, test_batch_ind = get_minibatches_idx('test', range(0, 20), 3, True)
    # print(train_batch_ind)
    # print(validate_batch_ind)
    # print(test_batch_ind)

    orch, piano, train_batch_ind, validate_batch_ind, test_batch_ind = load_data(data_path='../../Data/data.p', log_file_path='', temporal_order=4, minibatch_size=100, shuffle=True)
