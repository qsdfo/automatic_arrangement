#!/usr/bin/env python
# -*- coding: utf8 -*-

import random
import load_matrices
import LOP.Scripts.config
import avoid_tracks
import cPickle as pkl

def build_folds(store_folder, k_folds=10, temporal_order=20, batch_size=100, long_range_pred=1, num_max_contiguous_blocks=100,
    random_seed=None, logger_load=None):

    # Load files lists
    train_and_valid_files = pkl.load(open(store_folder + '/train_and_valid_files.pkl', 'rb'))
    train_only_files = pkl.load(open(store_folder + '/train_only_files.pkl', 'rb'))
    
    list_files_valid = train_and_valid_files.keys()
    list_files_train_only = train_only_files.keys()

    # Folds are built on files, not directly the indices
    # By doing so, we prevent the same file being spread over train, test and validate sets
    random.seed(random_seed)
    random.shuffle(list_files_valid)
    random.shuffle(list_files_train_only)

    if k_folds == -1:
        k_folds = len(list_files_valid)

    folds = []
    valid_names = []
    test_names = []

    # 1 Build the list of split_matrices
    for k in range(k_folds):
        # For each folds, build a list of train, valid, test blocks
        split_matrices_train=[]
        split_matrices_test=[]
        split_matrices_valid=[]
        # Keep track of files used for validating and testing
        this_valid_names=[]
        this_test_names=[]
            
        for counter, filename in enumerate(list_files_valid):
            counter_fold = counter + k
            if (counter_fold % k_folds) < k_folds-2:
                split_matrices_train.extend(train_and_valid_files[filename])
            elif (counter_fold % k_folds) == k_folds-2:
                this_valid_names.append(filename)
                split_matrices_valid.extend(train_and_valid_files[filename])
            elif (counter_fold % k_folds) == k_folds-1:
                this_test_names.append(filename)
                split_matrices_test.extend(train_and_valid_files[filename])

        for filename in list_files_train_only:
            split_matrices_train.extend(train_only_files[filename])
        
        folds.append(
            {"train": from_block_list_to_folds(split_matrices_train, temporal_order, batch_size, None, num_max_contiguous_blocks),
            "valid": from_block_list_to_folds(split_matrices_valid, temporal_order, batch_size, long_range_pred, num_max_contiguous_blocks),
            "test": from_block_list_to_folds(split_matrices_test, temporal_order, batch_size, long_range_pred, num_max_contiguous_blocks)}
        )
        valid_names.append(this_valid_names)
        test_names.append(this_test_names)

    return folds, valid_names, test_names


def from_block_list_to_folds(list_blocks, temporal_order, batch_size, long_range_pred, num_max_contiguous_blocks):
    # Shuffle the files lists
    random.shuffle(list_blocks)

    # Split them in blocks of size num_max_contiguous_blocks
    blocks = []
    counter = 0
    time = 0
    this_list_of_path = []
    this_list_of_valid_indices = []
    this_list_of_valid_indices_lr = []
    for block_folder in list_blocks:
        if counter > num_max_contiguous_blocks:
            this_dict = {       
                "batches": build_batches(this_list_of_valid_indices, batch_size),
                "chunks_folders": this_list_of_path
                }
            if long_range_pred:
                this_dict["batches_lr"] = build_batches(this_list_of_valid_indices_lr, batch_size)
            blocks.append(this_dict)
            counter = 0
            time = 0
            this_list_of_path = []
            this_list_of_valid_indices = []
            this_list_of_valid_indices_lr = []
        counter+=1
        
        # Update list of chunks
        this_list_of_path.append(block_folder)
        
        # Update the list of valid indices
        pr_piano, pr_orch, _ = load_matrices.load_matrix_NO_PROCESSING(block_folder, duration_piano_bool=False)
        duration = len(pr_piano)
        start_valid_ind = temporal_order - 1
        end_valid_ind = duration - temporal_order + 1
        this_indices = remove_silences(range(start_valid_ind, end_valid_ind), pr_piano, pr_orch)
        this_indices = [e+time for e in this_indices]
        this_list_of_valid_indices.extend(this_indices)
        if long_range_pred:
            end_valid_ind_lr = duration - temporal_order - long_range_pred + 1
            this_indices_lr = remove_silences(range(start_valid_ind, end_valid_ind_lr), pr_piano, pr_orch)
            this_indices_lr = [e+time for e in this_indices_lr]
            this_list_of_valid_indices_lr.extend(this_indices_lr)
        time += duration
    
    # Don't forget the last one !
    this_dict = {       
        "batches": build_batches(this_list_of_valid_indices, batch_size),
        "chunks_folders": this_list_of_path
        }
    if long_range_pred:
        this_dict["batches_lr"] = build_batches(this_list_of_valid_indices_lr, batch_size)
    blocks.append(this_dict)

    return blocks

def build_batches(ind, batch_size):
        batches = []
        position = 0
        n_ind = len(ind)
        
        n_batch = int(n_ind // batch_size)

        # Shuffle indices
        random.shuffle(ind)
       
        for i in range(n_batch):
            batches.append(ind[position:position+batch_size])
            position += batch_size
        # Smaller last batch
        if position < n_ind:
            batches.append(ind[position:n_ind])
        return batches

def remove_silences(indices, piano, orch):
    """ Remove silences from a set of indices. Remove both from piano and orchestra
    
    """
    flat_piano = piano.sum(axis=1)>0
    flat_orch = orch.sum(axis=1)>0
    flat_pr = flat_piano * flat_orch
    return [e for e in indices if (flat_pr[e] != 0)]

if __name__ == '__main__':
    build_folds("/Users/leo/Recherche/GitHub_Aciditeam/lop/Data_folds/Data__event_level8")