#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Normalization object
Created on Thu Nov 16 12:17:39 2017

@author: leo
"""


import numpy as np
import math
from load_matrices import load_matrices

class zero_mean_unit_variance(object):
    """Basically just a wrapper for scikit PCA
    """
    def __init__(self, train_folds, parameters):
        self.transformed_dim = None
        self.mean = 0
        self.var = 0
        # Init variables
        self.fit(train_folds, parameters)
        return

    def get_mean(self, train_folds, parameters):
        length = 0
        for path_matrix, indices in train_folds.iteritems():
            piano, _, _, _, _ = load_matrices(path_matrix, parameters)
            if self.transformed_dim is None:
                self.transformed_dim = piano.shape[1]
            flat_train_indices = [ind for batch in indices for ind in batch]
            mat_train = piano[flat_train_indices]
            self.mean += np.sum(mat_train, axis=0)
            length += mat_train.shape[0]
        self.mean /= length
        return
    
    def get_var(self, train_folds, parameters):
        length = 0
        for path_matrix, indices in train_folds.iteritems():
            piano, _, _, _, _ = load_matrices(path_matrix, parameters)
            flat_train_indices = [ind for batch in indices for ind in batch]
            mat_train = piano[flat_train_indices]
            self.var += np.sum(np.square(mat_train - self.mean), axis=0)
            length += mat_train.shape[0]
        self.var /= length
        return

    def fit(self, train_folds, parameters):
        self.get_mean(train_folds, parameters)
        self.get_var(train_folds, parameters)
        return

    def transform(self, matrix):
        return (matrix - self.mean) / math.sqrt(self.var)

# def apply_zca(piano, mean_piano, std_piano, zca_piano, epsilon):
#     piano_std = (piano-mean_piano) / (std_piano + epsilon)
#     piano = np.dot(piano_std, zca_piano)
#     return piano

# def get_whitening_mat(matrix, epsilon_std):
#     """Compute statistics for standardization and zca pre-processing
    
#     Parameters
#     ----------
#         matrix - (N,M) matrix where N is the number of samples and M the features dimension
#     """
#     # Normalize : Standardization + ZCA whitening
#     mean = np.mean(matrix, axis=0)
#     std = np.std(matrix, axis=0)
#     matrix_standard = (matrix - mean) / (std + epsilon_std)
# #    matrix_standard = (matrix - mean) 
    

    

#     def whitening(inputs, num_component_to_keep=None):
#         #Correlation matrix
#         sigma = np.dot(inputs.T, inputs)/inputs.shape[0]
        
    
#         #Singular Value Decomposition
#         U,S,V = np.linalg.svd(sigma)

#         # If you want to reduce the dimension of the data by keeping only the dimensions that explain variance the best
#         if num_component_to_keep:
#             U = U[:num_component_to_keep]
#             S = S[:num_component_to_keep]
    
#         #Whitening constant, it prevents division by zero
#         epsilon = 0.01
    
#         #ZCA Whitening matrix
#         PCAMatrix = np.dot(1.0/np.sqrt(np.diag(S) + epsilon), U.T)
#         ZCAMatrix = np.dot(U, PCAMatrix)
    
#         #Data whitening
#         return PCAMatrix, ZCAMatrix
#     pca, zca = whitening(matrix_standard)
#     return mean, std, pca, zca


# def normalize_data(piano, orch, train_indices_flat, parameters):
#     ## Normalize the data
#     piano_train = piano[train_indices_flat]
#     epsilon = 0.0001
#     if parameters["normalize"] == "standard_pca":
#         mean_piano, std_piano, pca_piano, _ = get_whitening_mat(piano_train, epsilon)
#         piano = apply_pca(piano, mean_piano, std_piano, pca_piano, epsilon) 
#         # Save the transformations for later
#         standard_pca_piano = {'mean_piano': mean_piano, 'std_piano': std_piano, 'pca_piano': pca_piano, 'epsilon': epsilon}
#         return piano, orch, standard_pca_piano
#     elif parameters["normalize"] == "standard_zca":
#         mean_piano, std_piano, _, zca_piano = get_whitening_mat(piano_train, epsilon)
#         piano = apply_zca(piano, mean_piano, std_piano, zca_piano, epsilon)
#         # Save the transformations for later
#         standard_zca_piano = {'mean_piano': mean_piano, 'std_piano': std_piano, 'zca_piano': zca_piano, 'epsilon': epsilon}
#         return piano, orch, standard_zca_piano
#     else:
#         raise Exception(str(parameters["normalize"]) + " is not a possible value for normalization parameter")