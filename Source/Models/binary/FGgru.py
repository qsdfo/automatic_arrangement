#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

# Model lop
from ..model_lop import Model_lop

# Hyperopt
from hyperopt import hp
from acidano.utils import hopt_wrapper
from math import log

# Numpy
import numpy as np

# Theano
import theano
import theano.tensor as T

# Performance measures
from acidano.utils.init import shared_normal, shared_zeros
from acidano.utils.measure import accuracy_measure, precision_measure, recall_measure
# Regularization
from acidano.utils.regularization import dropout_function
import acidano.utils.cost as cost_module
# Build matrix inputs
import acidano.utils.build_theano_input as build_theano_input


class FGgru(Model_lop):
    """LSTM multiple layers with regularization.

    Predictive model,
        input = piano(t)
        output = orchestra(t)
        measure = cross-entropy error function
            (output units are binary units (y_j) considered independent : i != j -> y_j indep y_i)
    """

    def __init__(self,
                 model_param,
                 dimensions,
                 checksum_database,
                 weights_initialization=None):

        Model_lop.__init__(self, model_param, dimensions, checksum_database)

        self.threshold = model_param['threshold']
        self.weighted_ce = model_param['weighted_ce']

        # Datas are represented like this:
        #   - visible = concatenation of the data : (num_batch, piano ^ orchestra_dim * temporal_order)
        self.n_p = dimensions['piano_dim']
        self.n_o = dimensions['orchestra_dim']

        # Number of hidden units
        self.n_hs = model_param['n_hidden']
        self.n_layer = len(self.n_hs)

        # Class normalization
        # Vector of size n_o, number of notes activation divided by mean number of note activation
        # See : https://arxiv.org/pdf/1703.10663.pdf
        self.class_normalization = model_param['class_normalization']
        self.mean_notes_activation = model_param['mean_notes_activation']

        self.W_z = {}
        self.U_z = {}
        self.b_z = {}
        self.W_r = {}
        self.U_r = {}
        self.b_r = {}
        self.W_h = {}
        self.U_h = {}
        self.b_h = {}

        if weights_initialization is None:
            # Weights
            for layer in xrange(self.n_layer):
                if layer == 0:
                    n_hlm1 = self.n_o
                else:
                    n_hlm1 = self.n_hs[layer-1]
                n_hl = self.n_hs[layer]
                # Forget gate
                self.U_z[layer] = shared_normal((n_hlm1, n_hl), 0.01, name='U_z'+str(layer))
                self.W_z[layer] = shared_normal((n_hl, n_hl), 0.01, name='W_z'+str(layer))
                self.b_z[layer] = shared_zeros((n_hl), name='b_z'+str(layer))
                # Reset gate
                self.U_r[layer] = shared_normal((n_hlm1, n_hl), 0.01, name='U_r'+str(layer))
                self.W_r[layer] = shared_normal((n_hl, n_hl), 0.01, name='W_r'+str(layer))
                self.b_r[layer] = shared_zeros((n_hl), name='b_r'+str(layer))
                # Recurence
                self.U_h[layer] = shared_normal((n_hlm1, n_hl), 0.01, name='U_h'+str(layer))
                self.W_h[layer] = shared_normal((n_hl, n_hl), 0.01, name='W_h'+str(layer))
                self.b_h[layer] = shared_zeros((n_hl), name='b_h'+str(layer))
            
            self.W_piano = shared_normal((self.n_p, self.n_hs[-1]), 0.01, name='W_piano')
            self.b_piano = shared_zeros((self.n_hs[-1]), name='b_piano')
            # Last predictive layer
            # self.W = shared_normal((self.n_hs[-1] * 2, self.n_o), 0.01, name='W')
            self.W = shared_normal((self.n_hs[-1], self.n_o), 0.01, name='W')
            self.b = shared_zeros((self.n_o), name='b')
            self.sum_coeff = theano.shared(1.0, name='sum_coeff')
        else:
            # Layer weights
            for layer, n_h_layer in enumerate(self.n_hs):
                self.W_z[layer] = weights_initialization['W_z'][layer]
                self.U_z[layer] = weights_initialization['U_z'][layer]
                self.b_z[layer] = weights_initialization['b_z'][layer]
                self.W_r[layer] = weights_initialization['W_r'][layer]
                self.U_r[layer] = weights_initialization['U_r'][layer]
                self.b_r[layer] = weights_initialization['b_r'][layer]
                self.W_h[layer] = weights_initialization['W_h'][layer]
                self.U_h[layer] = weights_initialization['U_h'][layer]
                self.b_h[layer] = weights_initialization['b_h'][layer]
            self.W_piano = weights_initialization['W_piano']
            self.b_piano = weights_initialization['b_piano']
            self.W = weights_initialization['W']
            self.b = weights_initialization['b']
            self.sum_coeff = weights_initialization['sum_coeff']

        self.params = self.W_z.values() + self.U_z.values() + self.b_z.values() + self.W_r.values() + self.U_r.values() +\
            self.b_r.values() + self.W_h.values() + self.U_h.values() + self.b_h.values() +\
            [self.W_piano, self.b_piano, self.W, self.b, self.sum_coeff]

        # Variables
        self.p = T.matrix('p', dtype=theano.config.floatX)
        self.o_past = T.tensor3('o_past', dtype=theano.config.floatX)
        self.o = T.matrix('o', dtype=theano.config.floatX)
        self.o_truth = T.matrix('o_truth', dtype=theano.config.floatX)
        self.p_gen = T.matrix('p_gen', dtype=theano.config.floatX)
        self.o_past_gen = T.tensor3('o_past_gen', dtype=theano.config.floatX)

        # Test values
        self.p.tag.test_value = np.random.randint(2, size=(self.batch_size, self.n_p)).astype(theano.config.floatX)
        self.o_past.tag.test_value = np.random.randint(2, size=(self.batch_size,  self.temporal_order-1, self.n_o)).astype(theano.config.floatX)
        self.o.tag.test_value = np.random.randint(2, size=(self.batch_size, self.n_o)).astype(theano.config.floatX)
        self.o_truth.tag.test_value = np.random.randint(2, size=(self.batch_size, self.n_o)).astype(theano.config.floatX)
        return

    ###############################
    #       STATIC METHODS
    #       FOR METADATA AND HPARAMS
    ###############################
    @staticmethod
    def get_hp_space():

        super_space = Model_lop.get_hp_space()

        space = {'n_hidden': hp.choice('n_hidden', [
            [hopt_wrapper.qloguniform_int('n_hidden_1_'+str(i), log(100), log(5000), 10) for i in range(1)],
            [hopt_wrapper.qloguniform_int('n_hidden_2_'+str(i), log(100), log(5000), 10) for i in range(2)],
            [hopt_wrapper.qloguniform_int('n_hidden_3_'+str(i), log(100), log(5000), 10) for i in range(3)],
            [hopt_wrapper.qloguniform_int('n_hidden_4_'+str(i), log(100), log(5000), 10) for i in range(4)]
        ]),
        }

        space.update(super_space)
        return space

    @staticmethod
    def name():
        return "FGgru"

    ###############################
    #  FORWARD PASS
    ###############################
    def corruption(self, h_lm1_t, n_lm1, axis):
        # Get input dimension (dieu que c'est moche)
        size_mask = (self.batch_size, n_lm1)
        # Apply dropout (see https://arxiv.org/pdf/1409.2329.pdf for details)
        # using a mask of zero
        if self.step_flag == 'train':
            h_lm1_t_corrupted = dropout_function(h_lm1_t, p_dropout=self.dropout_probability, size=size_mask, rng=self.rng)
        elif self.step_flag in ['validate', 'generate']:
            # Just multiply weights by the dropout ratio
            h_lm1_t_corrupted = h_lm1_t * (1-self.dropout_probability)
        else:
            raise ValueError("step_flag undefined")
        return h_lm1_t_corrupted

    def iteration(self, x_t, s_tm1,
                  W_z, U_z, b_z,
                  W_r, U_r, b_r,
                  W_h, U_h, b_h,
                  n_lm1):
        # Sum along last axis
        axis = x_t.ndim - 1
        # Dropout
        x_t_corrupted = self.corruption(x_t, n_lm1, axis)
        # Forget gate
        z = T.nnet.sigmoid(T.dot(x_t_corrupted, U_z) + T.dot(s_tm1, W_z) + b_z)
        # Reset gate
        r = T.nnet.sigmoid(T.dot(x_t_corrupted, U_r) + T.dot(s_tm1, W_r) + b_r)
        # Cell update term
        h = T.tanh(T.dot(x_t_corrupted, U_h) + T.dot(s_tm1 * r, W_h) + b_h)
        # Output gate
        s_t = (1-z)*h + z*s_tm1
        return s_t

    def forward_pass(self, orch_past, piano, batch_size):
        ################################################################
        ################################################################
        ################################################################
        # Normalization by the number of notes
        # orch_past_norm = self.number_note_normalization_fun(orch_past)
        # piano_norm = self.number_note_normalization_fun(piano)

        # TEST : batch norm on the input
        # orch_past_norm = batch_norm(orch_past, (self.temporal_order, self.n_o))
        # piano_norm = batch_norm(piano, (self.n_p,))
        #
        orch_past_norm = orch_past
        piano_norm = piano
        ################################################################
        ################################################################
        ################################################################

        # Time needs to be the first dimension
        orch_past_loop = orch_past_norm.dimshuffle((1, 0, 2))

        # Initialization
        input_layer = [None]*(self.n_layer+1)
        input_layer[0] = orch_past_loop
        n_lm1 = self.n_o

        # Loop
        for layer, n_h in enumerate(self.n_hs):
            s_0 = T.zeros((batch_size, n_h), dtype=theano.config.floatX)
            # Infer hidden states
            s_seq, updates = theano.scan(fn=self.iteration,
                                         sequences=[input_layer[layer]],
                                         outputs_info=[s_0],
                                         non_sequences=[self.W_z[layer], self.U_z[layer], self.b_z[layer],
                                                        self.W_r[layer], self.U_r[layer], self.b_r[layer],
                                                        self.W_h[layer], self.U_h[layer], self.b_h[layer],
                                                        n_lm1])

            # Inputs for the next layer are the hidden units of the current layer
            input_layer[layer+1] = s_seq
            # Update dimension
            n_lm1 = n_h

        # Last hidden units
        last_hidden = input_layer[self.n_layer]

        # Orchestra representation is the last state of the topmost rnn
        orchestra_repr = last_hidden[-1]

        ################################################################
        ################################################################
        ################################################################
        # Batch Normalization or no ??
        # orchestra_repr_norm = batch_norm(orchestra_repr, (n_lm1,))
        orchestra_repr_norm = orchestra_repr
        ################################################################
        ################################################################
        ################################################################
        
        ################################################################
        ################################################################
        # Piano through a mlp ?
        piano_repr = T.nnet.sigmoid(T.dot(piano_norm, self.W_piano) + self.b_piano)
        ################################################################
        ################################################################

        ################################################################
        ################################################################
        # Sum or concatenate
        # concat_input = T.concatenate([orchestra_repr_norm, piano_repr], axis=1)
        concat_input = orchestra_repr_norm + self.sum_coeff * piano_repr
        ################################################################
        ################################################################

        # Last layer
        orch_pred_mean = T.nnet.sigmoid(T.dot(concat_input, self.W) + self.b)

        ################################################################
        ################################################################
        ################################################################
        # Before sampling, we THRESHOLD
        orch_pred_mean_threshold = T.where(T.le(orch_pred_mean, self.threshold), 0, orch_pred_mean)
        ################################################################
        ################################################################
        ################################################################

        # Sampling
        orch_pred = self.rng.binomial(size=orch_pred_mean_threshold.shape, n=1, p=orch_pred_mean_threshold,
                                      dtype=theano.config.floatX)

        return orch_pred_mean, orch_pred_mean_threshold, orch_pred, updates

    ###############################
    #       COST
    ###############################
    def cost_updates(self, optimizer):
        # Infer Orchestra sequence
        pred, _, _, updates_train = self.forward_pass(self.o_past, self.p, self.batch_size)

        # Compute error function
        ############################
        ############################
        # TEST SEVERAL DIFFERENT COST FUNCTION
        if self.weighted_ce == 0:
            cost = T.nnet.binary_crossentropy(pred, self.o)  
        elif self.weighted_ce == 1:
            cost = cost_module.weighted_binary_cross_entropy_0(pred, self.o, self.class_normalization)
        elif self.weighted_ce == 2:
            cost = cost_module.weighted_binary_cross_entropy_1(pred, self.o, self.mean_notes_activation)
        elif self.weighted_ce == 3:
            cost = cost_module.weighted_binary_cross_entropy_2(pred, self.o)
        elif self.weighted_ce == 4:
            cost = cost_module.weighted_binary_cross_entropy_3(pred, self.o, self.mean_notes_activation)
        elif self.weighted_ce == 5:
            cost = cost_module.weighted_binary_cross_entropy_4(pred, self.o, self.class_normalization)
        
        # Sum over pitch axis
        cost = cost.sum(axis=1)
        # Mean along batch dimension
        cost = T.mean(cost)

        ########### BP MLL
        # cost, updates_bp_mll = cost_module.bp_mll(pred, self.o)
        # updates_train.update(updates_bp_mll)
        # cost = T.mean(cost)
        ############################
        ############################

        # Weight decay
        # cost = cost + self.weight_decay_coeff * self.get_weight_decay() + (0.1 * T.pow(self.b, 2).sum())
        cost = cost + self.weight_decay_coeff * self.get_weight_decay()
        # Monitor = cost
        monitor = cost
        # Update weights
        grads = T.grad(cost, self.params)
        updates_train = optimizer.get_updates(self.params, grads, updates_train)

        return cost, monitor, updates_train

    ###############################
    #       TRAIN FUNCTION
    ###############################
    def build_train_fn(self, optimizer, name):
        self.step_flag = 'train'
        # get the cost and the gradient corresponding to one step of CD-15
        cost, monitor, updates = self.cost_updates(optimizer)
        self.train_function = theano.function(inputs=[self.p, self.o_past, self.o],
                                outputs=[cost, monitor],
                                updates=updates,
                                name=name
                                )
        return


    def train_batch(self, batch_data):
        # Simply used for parsing the batch_data
        p, o_past, o = batch_data
        return self.train_function(p, o_past, o)

    ###############################
    #       PREDICTION
    ###############################
    def prediction_measure(self):
        # Generate the last frame for the sequence v
        _, o_pred_mean, _, updates_valid = self.forward_pass(self.o_past, self.p, self.batch_size)
        # Get the ground truth
        true_frame = self.o_truth
        # Measure the performances
        precision = precision_measure(true_frame, o_pred_mean)
        recall = recall_measure(true_frame, o_pred_mean)
        accuracy = accuracy_measure(true_frame, o_pred_mean)
        return precision, recall, accuracy, true_frame, self.o_past[:, -1, :], self.p, o_pred_mean, updates_valid

    ###############################
    #       VALIDATION FUNCTION
    ##############################
    def build_validation_fn(self, name):
        self.step_flag = 'validate'
        
        precision, recall, accuracy, true_frame, past_frame, piano_frame, predicted_frame, updates_valid = self.prediction_measure()

        self.validation_error = theano.function(inputs=[self.p, self.o_past, self.o_truth],
                                                outputs=[precision, recall, accuracy],
                                                updates=updates_valid,
                                                name=name
                                                )
        return

    def validate_batch(self, batch_data):
        # Simply used for parsing the batch_data
        p, o_past, o = batch_data
        return self.validation_error(p, o_past, o)

    ###############################
    #       GENERATION
    ###############################
    def get_generate_function(self, piano, orchestra, generation_length, seed_size, batch_generation_size, name="generate_sequence"):
        self.step_flag = 'generate'
        seed_size = self.temporal_order-1
        pred, _, next_orch, updates_next_sample = self.forward_pass(self.o_past_gen, self.p_gen, batch_generation_size)
        # Compile a function to get the next visible sample

        next_sample = theano.function(
            inputs=[self.o_past_gen, self.p_gen],
            outputs=[next_orch, pred],
            updates=updates_next_sample,
            name="next_sample",
        )

        def closure(ind):
            # Initialize generation matrice
            piano_gen, orchestra_gen = build_theano_input.initialization_generation(piano, orchestra, ind, generation_length, batch_generation_size, seed_size)
            for index in xrange(seed_size, generation_length, 1):
                # Build past vector
                p_gen = piano_gen[:, index, :]
                # Automatically map a silence to a silence
                if p_gen.sum() == 0:
                    o_t = np.zeros((self.n_o,))
                else:
                    o_past_gen = orchestra_gen[:, index-self.temporal_order+1:index, :]
                    # Get the next sample
                    out_theano = next_sample(o_past_gen, p_gen)
                    o_t = (np.asarray(out_theano)[0]).astype(theano.config.floatX)
                # Add this visible sample to the generated orchestra
                orchestra_gen[:, index, :] = o_t
            return (orchestra_gen,)

        return closure

    @staticmethod
    def get_static_config():
        model_space = {}
        model_space['batch_size'] = 200
        model_space['temporal_order'] = 10
        model_space['dropout_probability'] = 0
        model_space['weight_decay_coeff'] = 1e-4
        # model_space['number_note_normalization'] = True
        # Last layer could be of size piano = 93
        model_space['n_hidden'] = [500, 500, 100]
        return model_space

    def generator(self, piano, orchestra, index):    
        p = piano[index, :]
        o_past = build_theano_input.build_sequence(orchestra, index-1, self.batch_size, self.temporal_order-1, self.n_o)
        o_truth = orchestra[index, :]
        return p, o_past, o_truth
