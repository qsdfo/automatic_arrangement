#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Module for normalization pre-processing on training data
Created on Thu Nov 16 12:17:39 2017

@author: leo
"""

import numpy as np


def apply_pca(piano, mean_piano, std_piano, pca_piano, epsilon):
    piano_std = (piano-mean_piano) / (std_piano + epsilon)
    piano = np.dot(piano_std, pca_piano)
    return piano

def apply_zca(piano, mean_piano, std_piano, zca_piano, epsilon):
    piano_std = (piano-mean_piano) / (std_piano + epsilon)
    piano = np.dot(piano_std, zca_piano)
    return piano

def get_whitening_mat(matrix, epsilon_std):
    """Compute statistics for standardization and zca pre-processing
    
    Parameters
    ----------
        matrix - (N,M) matrix where N is the number of samples and M the features dimension
    """
    # Normalize : Standardization + ZCA whitening
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    matrix_standard = (matrix - mean) / (std + epsilon_std)
#    matrix_standard = (matrix - mean) 
    
    def whitening(inputs, num_component_to_keep=None):
        #Correlation matrix
        sigma = np.dot(inputs.T, inputs)/inputs.shape[0]
        
    
        #Singular Value Decomposition
        U,S,V = np.linalg.svd(sigma)

        # If you want to reduce the dimension of the data by keeping only the dimensions that explain variance the best
        if num_component_to_keep:
            U = U[:num_component_to_keep]
            S = S[:num_component_to_keep]
    
        #Whitening constant, it prevents division by zero
        epsilon = 0.01
    
        #ZCA Whitening matrix
        PCAMatrix = np.dot(1.0/np.sqrt(np.diag(S) + epsilon), U.T)
        ZCAMatrix = np.dot(U, PCAMatrix)
    
        #Data whitening
        return PCAMatrix, ZCAMatrix
    pca, zca = whitening(matrix_standard)
    return mean, std, pca, zca