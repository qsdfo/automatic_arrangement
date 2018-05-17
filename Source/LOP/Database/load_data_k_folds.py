#!/usr/bin/env python
# -*- coding: utf8 -*-

import random
import load_matrices
import LOP.Scripts.config
import LOP.Database.avoid_tracks
import pickle as pkl

def build_folds(store_folder, k_folds=10, temporal_order=20, train_batch_size=100, long_range_pred=1, training_mode=0, num_max_contiguous_blocks=100,
    random_seed=None, logger_load=None):

    # Load files lists
    train_and_valid_files = pkl.load(open(store_folder + '/train_and_valid_files.pkl', 'rb'))
    train_only_files = pkl.load(open(store_folder + '/train_only_files.pkl', 'rb'))
    
    # Pretraining files
    pre_train_and_valid_files = pkl.load(open(store_folder + '/train_and_valid_files_pretraining.pkl', 'rb'))
    pre_train_only_files = pkl.load(open(store_folder + '/train_only_files_pretraining.pkl', 'rb'))
    
    list_files_valid = list(train_and_valid_files.keys())
    list_files_train_only = list(train_only_files.keys())
    pre_list_files_valid = list(pre_train_and_valid_files.keys())
    pre_list_files_train_only = list(pre_train_only_files.keys())

    # Folds are built on files, not directly the indices
    # By doing so, we prevent the same file being spread over train, test and validate sets
    random.seed(random_seed)
    random.shuffle(list_files_valid)
    random.shuffle(list_files_train_only)
    random.shuffle(pre_list_files_valid)
    random.shuffle(pre_list_files_train_only)

    if k_folds == -1:
        k_folds = len(list_files_valid)

    folds = []
    pretraining_folds = []
    valid_names = []
    test_names = []

    if training_mode==0:
        # Pre-train then train
        pretraining_fold, _, _ = build_one_fold(0, k_folds, pre_list_files_valid, pre_list_files_train_only, 
            pre_train_and_valid_files, pre_train_only_files, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks)
        pretraining_folds = [pretraining_fold]
    elif training_mode==1:
        # Pretraining files are added only to the training set of files
        train_only_files.update(pre_train_and_valid_files)
        train_only_files.update(pre_train_only_files)
        list_files_train_only = list(train_only_files.keys())
        # Reshuffle
        random.shuffle(list_files_train_only)

    # Build the list of split_matrices
    for k in range(k_folds):
        if training_mode==0:
            one_fold, this_valid_names, this_test_names = build_one_fold(k, k_folds, list_files_valid, list_files_train_only, 
                train_and_valid_files, train_only_files, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks)
        elif training_mode==1:
            one_fold, this_valid_names, this_test_names = build_one_fold(k, k_folds, list_files_valid, list_files_train_only, 
                train_and_valid_files, train_only_files, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks)
        
        folds.append(one_fold)
        valid_names.append(this_valid_names)
        test_names.append(this_test_names)

    return folds, pretraining_folds, valid_names, test_names


def build_one_fold(k, k_folds, list_files_valid, list_files_train_only, train_and_valid_files, train_only_files, 
    temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks):
    split_matrices_train=[]
    split_matrices_test=[]
    split_matrices_valid=[]
    # Keep track of files used for validating and testing
    this_valid_names=[]
    this_test_names=[]

    valid_batch_size = 10000 # Still se batches to avoid OOM errors
        
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
    
    this_fold = {"train": from_block_list_to_folds(split_matrices_train, temporal_order, train_batch_size, None, num_max_contiguous_blocks),
        "valid": from_block_list_to_folds(split_matrices_valid, temporal_order, valid_batch_size, long_range_pred, num_max_contiguous_blocks),
        "test": from_block_list_to_folds(split_matrices_test, temporal_order, valid_batch_size, long_range_pred, num_max_contiguous_blocks)}
    return this_fold, this_valid_names, this_test_names


def from_block_list_to_folds(list_blocks, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks):
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
                "batches": build_batches(this_list_of_valid_indices, train_batch_size),
                "chunks_folders": this_list_of_path
                }
            if long_range_pred:
                this_dict["batches_lr"] = build_batches(this_list_of_valid_indices_lr, train_batch_size)
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
        "batches": build_batches(this_list_of_valid_indices, train_batch_size),
        "chunks_folders": this_list_of_path
        }
    if long_range_pred:
        this_dict["batches_lr"] = build_batches(this_list_of_valid_indices_lr, train_batch_size)
    blocks.append(this_dict)

    return blocks

def build_batches(ind, train_batch_size):
        batches = []
        position = 0
        n_ind = len(ind)

        if train_batch_size:
            n_batch = int(n_ind // train_batch_size)
            # Shuffle indices
            random.shuffle(ind)
        else:
            # One batch for valid and test, 
            # and don't shuffle (useless)
            n_batch = 1
            train_batch_size = n_ind
        
        for i in range(n_batch):
            batches.append(ind[position:position+train_batch_size])
            position += train_batch_size
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