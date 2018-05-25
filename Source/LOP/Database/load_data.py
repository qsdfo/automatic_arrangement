#!/usr/bin/env python
# -*- coding: utf8 -*-

import random
import load_matrices
import LOP.Scripts.config
import LOP.Database.avoid_tracks
import pickle as pkl

def build_one_fold(k, k_folds, t_dict, tv_dict, tvt_dict, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks, random_inst):
    split_matrices_train=[]
    split_matrices_test=[]
    split_matrices_valid=[]
    # Keep track of files used for validating and testing
    this_train_names=[]
    this_valid_names=[]
    this_test_names=[]

    for counter, filename in enumerate(tvt_dict.keys()):
        counter_fold = counter + k
        if (counter_fold % k_folds) < k_folds-2:
            this_train_names.append(filename)
            split_matrices_train.extend(tvt_dict[filename])
        elif (counter_fold % k_folds) == k_folds-2:
            this_valid_names.append(filename)
            split_matrices_valid.extend(tvt_dict[filename])
        elif (counter_fold % k_folds) == k_folds-1:
            this_test_names.append(filename)
            split_matrices_test.extend(tvt_dict[filename])

    for counter, filename in enumerate(tv_dict.keys()):
        counter_fold = counter + k
        if (counter_fold % k_folds) < k_folds-2:
            this_train_names.append(filename)
            split_matrices_train.extend(tv_dict[filename])
        elif (counter_fold % k_folds) == k_folds-1:
            this_valid_names.append(filename)
            split_matrices_valid.extend(tv_dict[filename])

    for filename in t_dict.keys():
        this_train_names.append(filename)
        split_matrices_train.extend(t_dict[filename])
    
    this_fold = {"train": from_block_list_to_folds(split_matrices_train, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks, random_inst),
        "valid": from_block_list_to_folds(split_matrices_valid, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks, random_inst),
        "test": from_block_list_to_folds(split_matrices_test, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks, random_inst)}
    return this_fold, this_train_names, this_valid_names, this_test_names


def from_block_list_to_folds(list_blocks, temporal_order, train_batch_size, long_range_pred, num_max_contiguous_blocks, random_inst):
    # Shuffle the files lists
    random_inst.shuffle(list_blocks)

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
        pr_piano, _, pr_orch, _, _ = load_matrices.load_matrix_NO_PROCESSING(block_folder, duration_piano_bool=False, mask_orch_bool=False)
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
        "batches": build_batches(this_list_of_valid_indices, train_batch_size, random_inst),
        "chunks_folders": this_list_of_path
        }
    if long_range_pred:
        this_dict["batches_lr"] = build_batches(this_list_of_valid_indices_lr, train_batch_size, random_inst)
    blocks.append(this_dict)

    return blocks

def build_batches(ind, train_batch_size, random_inst):
        batches = []
        position = 0
        n_ind = len(ind)

        if train_batch_size:
            n_batch = int(n_ind // train_batch_size)
            # Shuffle indices
            random_inst.shuffle(ind)
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