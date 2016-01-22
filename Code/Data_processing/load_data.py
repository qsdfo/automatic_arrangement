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
    log_file = open(log_file_path, "wb")
    log_file.write("### LOADING DATA ###\n\n")
    log_file.write("## Unpickle data... ")

    data_all = cPickle.load(open(data_path, "rb"))
    log_file.write("Done !\n\n")

    quantization = data_all['quantization']
    log_file.write("## Quantization : %d\n\n" % quantization)

    # Build the pianoroll and valid indexes
    log_file.write("## Reading scores :\n")
    scores = data_all['scores']
    instru_mapping = data_all['instru_mapping']
    orchestra_dimension = data_all['orchestra_dimension']
    last_stop = 0
    first_writting = True
    start_time = 0
    end_time = 0
    valid_index = []
    import pdb; pdb.set_trace()
    for info in scores.itervalues():
        # Concatenate pianoroll along time dimension
        for instru, pianoroll_instru in info['pianoroll'].iteritems():
            if 'data' in locals():
                if first_writting:
                    # set start and end time
                    start_time += end_time
                    end_time = start_time + np.shape(pianoroll_instru)[0]
                    # concatenate empty pianoroll
                    empty_data = np.empty([end_time - start_time, orchestra_dimension])
                    data = np.concatenate(data, empty_data, axis=0)
            else:
                end_time = start_time + np.shape(pianoroll_instru)[0]
                data = np.empty([end_time, orchestra_dimension])
            instru_ind_start = instru_mapping[instru][0]
            instru_ind_end = instru_mapping[instru][1]
            data[start_time:end_time, instru_ind_start:instru_ind_end] = pianoroll_instru
            first_writting = False
        # Valid indexes
        N_pr = data.shape[0]
        valid_index += range(last_stop + temporal_order, last_stop + N_pr)
        last_stop = last_stop + N_pr

        first_writting = True

        log_file.write("# Score %s : " % info['filename'])

    data_shared = theano.shared(np.asarray(data, dtype=theano.config.floatX))

    log_file.close()
    return data_shared, valid_index, quantization


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

    n = len(idx_list)

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

    return minibatches_train, minibatches_validate, minibatches_test


def load_data(data_path, log_file_path, temporal_order, minibatch_size, shuffle=False, split=(0.7, 0.1, 0.2)):
    data, valid_index, quantization = get_data(data_path, log_file_path, temporal_order)
    train_index, validate_index, test_index = get_minibatches_idx(log_file_path, valid_index, minibatch_size, shuffle, split)
    return data, train_index, validate_index, test_index


if __name__ == '__main__':
    # train_batch_ind, validate_batch_ind, test_batch_ind = get_minibatches_idx(range(0, 20), 3, True)
    # print(train_batch_ind.get_value())
    # print(validate_batch_ind.get_value())
    # print(test_batch_ind.get_value())

    minibatches, train_batch_ind, validate_batch_ind, test_batch_ind = load_data(data_path='../../Data/data.p', log_file_path='log_file.txt', temporal_order=4, minibatch_size=100, shuffle=True)
