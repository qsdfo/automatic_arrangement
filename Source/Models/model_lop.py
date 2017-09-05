#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

# Plot lib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from acidano.visualization.numpy_array.visualize_numpy import visualize_mat

from acidano.utils import hopt_wrapper

import numpy as np
from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import theano.tensor as T

from bokeh.plotting import figure, output_file, save

from hyperopt import hp
from math import log

from abc import ABCMeta, abstractmethod


class Model_lop(object):
    """
    Template class for the lop models.

    Contains plot methods
    """

    __metaclass__ = ABCMeta

    def __init__(self, model_param, dimensions, checksum_database):
        # Training parameters
        self.batch_size = dimensions['batch_size']
        self.temporal_order = dimensions['temporal_order']

        # Regularization paramters
        self.dropout_probability = model_param['dropout_probability']
        self.weight_decay_coeff = model_param['weight_decay_coeff']

        # Do we normalize input ?
        # self.number_note_normalization = model_param['number_note_normalization']

        # Numpy and theano random generators
        self.rng_np = RandomState(25)
        self.rng = RandomStreams(seed=25)

        # Database checksums
        self.checksum_database = checksum_database

        self.params = []
        self.step_flag = None
        return

    @staticmethod
    def get_hp_space():
        space_training = {'batch_size': hopt_wrapper.quniform_int('batch_size', 50, 500, 1),
                          'temporal_order': hopt_wrapper.qloguniform_int('temporal_order', log(3), log(20), 1)
                          }

        space_regularization = {'dropout_probability': hp.choice('dropout', [
            0.0,
            hp.normal('dropout_probability', 0.5, 0.1)
        ]),
            'weight_decay_coeff': hp.choice('weight_decay_coeff', [
                0.0,
                hp.uniform('a', 1e-4, 1e-4)
            ]),
        }

        space_training.update(space_regularization)
        return space_training


    ###############################
    # Abstract methods for the 
    # structure of the train function
    ###############################
    @abstractmethod
    def build_train_fn(self, optimizer, name):
        pass

    @abstractmethod
    def train_batch(self, batch_data):
        pass

    @abstractmethod
    def build_validation_fn(self, name):
        pass

    @abstractmethod
    def validate_batch(self, batch_data):
        pass

    @abstractmethod
    def generator(self, piano, orchestra, indices):
        pass

    ###############################
    # Set flags for the different steps
    ###############################
    def get_train_function(self):
        self.step_flag = 'train'
        return

    def get_validation_error(self):
        self.step_flag = 'validate'
        return

    def get_generate_function(self):
        self.step_flag = 'generate'
        return

    def save_weights(self, save_folder):
        def plot_process(param_shared):
            param = param_shared.get_value()

            # temp_csv = save_folder + '/' + param_shared.name + '.csv'
            # np.savetxt(temp_csv, param, delimiter=',')

            # Get mean, std and write title
            mean = np.mean(param)
            std = np.std(param)
            min_v = np.min(param)
            max_v = np.max(param)
            title = param_shared.name + " mean = " + str(mean) + " std = " + str(std) +\
                "\nmin = " + str(min_v) + " max = " + str(max_v)

            # Plot histogram
            fig = plt.figure()
            fig.suptitle(title, fontsize=14, fontweight='bold')

            ax = fig.add_subplot(111)
            fig.subplots_adjust(top=0.85)

            ax.set_xlabel('value')
            ax.set_ylabel('nb_occurence')

            param_ravel = param.ravel()
            # Check for NaN values
            if np.sum(np.isnan(param_ravel)):
                # Give an arbitrary value
                param_ravel = np.zeros(param_ravel.shape) - 1
                fig.suptitle(title + " NAN !!", fontsize=14, fontweight='bold')

            n, bins, patches = plt.hist(param_ravel, bins=50, normed=1, histtype='bar', rwidth=0.8)
            plt.savefig(save_folder + '/' + param_shared.name + '.pdf')
            plt.close(fig)

            # D3js plot (heavy...)
            temp_csv = save_folder + '/' + param_shared.name + '.csv'
            np.savetxt(temp_csv, param, delimiter=',')
            visualize_mat(param, save_folder, param_shared.name)

            # Plot matrices
            xdim = param.shape[0]
            if len(param.shape) == 1:
                param = param.reshape((xdim,1))
            ydim = param.shape[1]
            minparam = param.min()
            maxparam = param.max()
            # Avoid division by zero
            if minparam == maxparam:
                maxparam = minparam + 1
            view = param.view(dtype=np.uint8).reshape((xdim, ydim, 4))
            for i in range(xdim):
                for j in range(ydim):
                    value = (param[i][j] - minparam) / (maxparam - minparam)
                    view[i, j, 0] = int(255 * value)
                    view[i, j, 1] = int(255 * value)
                    view[i, j, 2] = int(255 * value)
                    view[i, j, 3] = 255
            output_file(save_folder + '/' + param_shared.name + '_bokeh.html')
            p = figure(title=param_shared.name, x_range=(0, xdim), y_range=(0, ydim))
            p.image_rgba(image=[param.T], x=[0], y=[0], dw=[xdim], dh=[ydim])
            save(p)

        # Plot weights
        for param_shared in self.params:
            if not (param_shared.name == 'sum_coeff'):
                plot_process(param_shared)

    def get_weight_decay(self):
        ret = 0
        for param in self.params:
            ret += T.pow(param, 2).sum()
        return ret