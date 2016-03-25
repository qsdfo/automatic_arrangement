#!/usr/bin/env python
# -*- coding: utf8 -*-

# Based on Bengio's team implementation

# Numpy
import numpy as np
# Theano
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
# Utils
from Models.Utils.init import shared_normal, shared_zeros
from Models.Utils.forward import propup_linear, propup_sigmoid, propup_relu, propup_tanh, propup_softplus
# Optimize
from Optim import adam_L2
# Performance measures
from Measure.accuracy_measure import accuracy_measure
from Measure.precision_measure import precision_measure
from Measure.recall_measure import recall_measure


# Cost functions
def Gaussian(y, mu, sig):
    """
    Gaussian negative log-likelihood
    Parameters
    ----------
    y   : TensorVariable
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    # Expression ok :
    #   -log(p(x))
    # with p a gaussian
    # BUT NOT WITH TEST VALUES
    nll = 0.5 * T.sum(T.sqr(y - mu) / sig ** 2 + 2 * T.log(sig) +
                      T.log(2 * np.pi), axis=1)

    # Summed over input dimension
    return nll


def KLGaussianGaussian(mu1, sig1, mu2, sig2):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and Gaussian dist.
    Parameters
    ----------
    mu1  : FullyConnected (Linear)
    sig1 : FullyConnected (Softplus)
    mu2  : FullyConnected (Linear)
    sig2 : FullyConnected (Softplus)
    """
    # Checked : formula ok
    # NOT CHECKED WITH TEST_VALUES
    kl = 0.5 * (2 * T.log(sig2)
                - 2 * T.log(sig1)
                + (sig1 ** 2 + (mu1 - mu2) ** 2) / sig2 ** 2
                - 1)
    return kl


class Variational_LSTM(object):
    '''Simple class to train an RNN-RBM from MIDI files and to generate sample
    sequences.'''

    def __init__(self,
                 orch=None,  # sequences as Theano matrices
                 piano=None,  # sequences as Theano matrices
                 units_dim=None,
                 reparametrization_dim=None,
                 lstm_dim=None,
                 weights=(None,) * 12,
                 numpy_rng=None,
                 theano_rng=None):
        '''Constructs and compiles Theano functions for training and sequence
        generation.
        n_hidden : integer
            Number of hidden units of the conditional RBMs.
        n_hidden_recurrent : integer
            Number of hidden units of the RNN.
        lr : float
            Learning rate
        r : (integer, integer) tuple
            Specifies the pitch range of the piano-roll in MIDI note numbers,
            including r[0] but not r[1], such that r[1]-r[0] is the number of
            visible units of the RBM at a given time step. The default (21,
            109) corresponds to the full range of piano (88 notes).
        dt : float
            Sampling period when converting the MIDI files into piano-rolls, or
            equivalently the time difference between consecutive time steps.'''

        # Random generators
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        # initialize input layer for standalone CRBM or layer0 of CDBN
        self.orch = orch
        if not orch:
            self.orch = T.matrix('orch')

        self.piano = piano
        if not piano:
            self.piano = T.matrix('piano')

        # units_dim can't be None
        assert (units_dim is not None), "You should provide dimensions for the units in the net"
        piano_dim = units_dim['piano_dim']
        orch_dim = units_dim['orch_dim']
        h_dim = units_dim['h_dim']
        z_dim = units_dim['z_dim']
        self.h_dim = h_dim

        if reparametrization_dim is None:
            p2h_dim = 200
            o2h_dim = 200
            z2h_dim = 200
            prior2z_dim = 200
        else:
            p2h_dim = reparametrization_dim['p2h_dim']
            o2h_dim = reparametrization_dim['o2h_dim']
            z2h_dim = reparametrization_dim['z2h_dim']
            prior2z_dim = reparametrization_dim['prior2z_dim']

        if reparametrization_dim is None:
            input_dim = 200
            cell_dim = 200
            forget_dim = 200
        else:
            input_dim = lstm_dim['input_dim']
            cell_dim = lstm_dim['cell_dim']
            forget_dim = lstm_dim['forget_dim']
        self.cell_dim = cell_dim

        # Reparametrization networks
        # piano network
        self.W_p1 = shared_normal(piano_dim, p2h_dim, 0.001)
        self.W_p2 = shared_normal(p2h_dim, p2h_dim, 0.001)
        self.W_p3 = shared_normal(p2h_dim, p2h_dim, 0.001)
        self.W_p4 = shared_normal(p2h_dim, p2h_dim, 0.001)
        self.b_p1 = shared_zeros(p2h_dim)
        self.b_p2 = shared_zeros(p2h_dim)
        self.b_p3 = shared_zeros(p2h_dim)
        self.b_p4 = shared_zeros(p2h_dim)

        # orchestra network
        self.W_o1 = shared_normal(orch_dim, o2h_dim, 0.001)
        self.W_o2 = shared_normal(o2h_dim, o2h_dim, 0.001)
        self.W_o3 = shared_normal(o2h_dim, o2h_dim, 0.001)
        self.W_o4 = shared_normal(o2h_dim, o2h_dim, 0.001)
        self.b_o1 = shared_zeros(o2h_dim)
        self.b_o2 = shared_zeros(o2h_dim)
        self.b_o3 = shared_zeros(o2h_dim)
        self.b_o4 = shared_zeros(o2h_dim)

        # latent network
        self.W_z1 = shared_normal(z_dim, z2h_dim, 0.001)
        self.W_z2 = shared_normal(z2h_dim, z2h_dim, 0.001)
        self.W_z3 = shared_normal(z2h_dim, z2h_dim, 0.001)
        self.W_z4 = shared_normal(z2h_dim, z2h_dim, 0.001)
        self.b_z1 = shared_zeros(z2h_dim)
        self.b_z2 = shared_zeros(z2h_dim)
        self.b_z3 = shared_zeros(z2h_dim)
        self.b_z4 = shared_zeros(z2h_dim)

        # Encoder (inference)
        enc_dim = o2h_dim + p2h_dim + h_dim
        self.W_enc1 = shared_normal(enc_dim, enc_dim, 0.001)
        self.W_enc2 = shared_normal(enc_dim, enc_dim, 0.001)
        self.W_enc3 = shared_normal(enc_dim, enc_dim, 0.001)
        self.W_enc4 = shared_normal(enc_dim, enc_dim, 0.001)
        self.W_enc_mu = shared_normal(enc_dim, z_dim, 0.001)
        self.W_enc_sig = shared_normal(enc_dim, z_dim, 0.001)
        self.b_enc1 = shared_zeros(enc_dim)
        self.b_enc2 = shared_zeros(enc_dim)
        self.b_enc3 = shared_zeros(enc_dim)
        self.b_enc4 = shared_zeros(enc_dim)
        self.b_enc_mu = shared_zeros(z_dim)
        self.b_enc_sig = shared_zeros(z_dim)

        # Decoder (generation)
        # prior
        prior_dim = p2h_dim + h_dim
        self.W_prior1 = shared_normal(prior_dim, prior2z_dim, 0.001)
        self.W_prior2 = shared_normal(prior2z_dim, prior2z_dim, 0.001)
        self.W_prior3 = shared_normal(prior2z_dim, prior2z_dim, 0.001)
        self.W_prior4 = shared_normal(prior2z_dim, prior2z_dim, 0.001)
        self.W_prior_mu = shared_normal(prior2z_dim, z_dim, 0.001)
        self.W_prior_sigma = shared_normal(prior2z_dim, z_dim, 0.001)
        self.b_prior1 = shared_zeros(prior_dim)
        self.b_prior2 = shared_zeros(prior_dim)
        self.b_prior3 = shared_zeros(prior_dim)
        self.b_prior4 = shared_zeros(prior_dim)
        self.b_prior_mu = shared_zeros(z_dim)
        self.b_prior_sigma = shared_zeros(z_dim)

        dec_dim = z2h_dim + p2h_dim + h_dim
        self.W_dec1 = shared_normal(dec_dim, dec_dim, 0.001)
        self.W_dec2 = shared_normal(dec_dim, dec_dim, 0.001)
        self.W_dec3 = shared_normal(dec_dim, dec_dim, 0.001)
        self.W_dec4 = shared_normal(dec_dim, dec_dim, 0.001)
        self.W_dec_mu = shared_normal(dec_dim, orch_dim, 0.001)
        self.W_dec_sigma = shared_normal(dec_dim, orch_dim, 0.001)
        self.b_dec1 = shared_zeros(dec_dim)
        self.b_dec2 = shared_zeros(dec_dim)
        self.b_dec3 = shared_zeros(dec_dim)
        self.b_dec4 = shared_zeros(dec_dim)
        self.b_dec_mu = shared_zeros(orch_dim)
        self.b_dec_sigma = shared_zeros(orch_dim)

        # Recurence function
        # LSTM
        # input gate
        LSTM_in_dim = o2h_dim + z2h_dim
        self.L_oi = shared_normal(LSTM_in_dim, input_dim, 0.001)
        self.L_hi = shared_normal(h_dim, input_dim, 0.001)
        self.b_i = shared_zeros(input_dim)
        # Internal cell
        self.L_oc = shared_normal(LSTM_in_dim, cell_dim, 0.001)
        self.L_hc = shared_normal(h_dim, cell_dim, 0.001)
        self.b_i = shared_zeros(cell_dim)
        # Forget gate

        self.L_of = shared_normal(LSTM_in_dim, forget_dim, 0.001)
        self.L_hf = shared_normal(h_dim, forget_dim, 0.001)
        self.b_f = shared_zeros(forget_dim)
        # Output
        # No L_cout... as in Theano tuto
        self.L_oout = shared_normal(LSTM_in_dim, h_dim, 0.001)
        self.L_hout = shared_normal(h_dim, h_dim, 0.001)
        self.b_out = shared_zeros(h_dim)

        # We don't use the same learning rate for the different parts of the network
        # Hence we group them in different variables
        self.params = self.W_p1, self.W_p2, self.W_p3, self.W_p4, \
            self.b_p1, self.b_p2, self.b_p3, self.b_p4, \
            self.W_o1, self.W_o2, self.W_o3, self.W_o4, \
            self.b_o1, self.b_o2, self.b_o3, self.b_o4, \
            self.W_z1, self.W_z2, self.W_z3, self.W_z4, \
            self.b_z1, self.b_z2, self.b_z3, self.b_z4, \
            self.W_enc1, self.W_enc2, self.W_enc3, self.W_enc4, \
            self.W_enc_mu, self.W_enc_sig, \
            self.b_enc1, self.b_enc2, self.b_enc3, self.b_enc4, \
            self.b_enc_mu, self.b_enc_sig, \
            self.W_prior1, self.W_prior2, self.W_prior3, self.W_prior4, \
            self.W_prior_mu, self.W_prior_sigma, \
            self.b_prior1, self.b_prior2, self.b_prior3, self.b_prior4, \
            self.b_prior_mu, self.b_prior_sigma, \
            self.W_dec1, self.W_dec2, self.W_dec3, self.W_dec4, \
            self.W_dec_mu, self.W_dec_sigma, \
            self.b_dec1, self.b_dec2, self.b_dec3, self.b_dec4, \
            self.b_dec_mu, self.b_dec_sigma, \
            self.L_oi, self.L_hi, self.b_i, \
            self.L_oc, self.L_hc, self.b_i, \
            self.L_of, self.L_hf, self.b_f, \
            self.L_oout, self.L_hout, self.b_out

        # Initialize the optimizer
        self.optimizer = adam_L2(b1=0.9, b2=0.999, alpha=0.001, epsilon=1e-8)

    def Gaussian_sample(self, mu, sig):

        epsilon = self.theano_rng.normal(size=(mu.shape),
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        z = mu + sig * epsilon
        return z

    def lstm_prop(self, o_t, c_tm1, h_tm1):
        # Input gate
        i = propup_sigmoid([o_t, h_tm1], [self.L_oi, self.L_hi], self.b_i)
        # Forget gate
        f = propup_sigmoid([o_t, h_tm1], [self.L_of, self.L_hf], self.b_f)
        # Cell update term
        c_tilde = propup_tanh([o_t, h_tm1], [self.L_oc, self.L_hc], self.b_c)
        c_t = f * c_tm1 + i * c_tilde
        # Output gate
        o = propup_sigmoid([o_t, h_tm1], [self.L_oout, self.L_hout], self.b_out)
        # h_t
        h_t = o * T.tanh(c_t)

        return h_t, c_t

    def inference(self, o, p, h):
        # Infering z_t sequence from orch_t and piano_t
        #   (under both prior AND q distribution)

        # Initialize h_0 and c_0 states
        h_0 = T.zeros((self.h_dim,))
        c_0 = T.zeros((self.cell_dim,))

        # Orch network
        o_1 = propup_relu(o, self.W_o1, self.b_o1)
        o_2 = propup_relu(o_1, self.W_o2, self.b_o2)
        o_3 = propup_relu(o_2, self.W_o3, self.b_o3)
        o_4 = propup_relu(o_3, self.W_o4, self.b_o4)

        # Piano network
        p_1 = propup_relu(p, self.W_p1, self.b_p1)
        p_2 = propup_relu(p_1, self.W_p2, self.b_p2)
        p_3 = propup_relu(p_2, self.W_p3, self.b_p3)
        p_4 = propup_relu(p_3, self.W_p4, self.b_p4)

        def inner_fn(o_t, p_t, h_tm1, c_tm1):
            # This inner function describes one step of the recurrent process
            # Prior
            prior_1_t = propup_relu([p_t, h_tm1], self.W_prior1, self.b_prior1)
            prior_2_t = propup_relu(prior_1_t, self.W_prior2, self.b_prior2)
            prior_3_t = propup_relu(prior_2_t, self.W_prior3, self.b_prior3)
            prior_4_t = propup_relu(prior_3_t, self.W_prior4, self.b_prior4)
            prior_mu_t = propup_linear(prior_4_t, self.W_prior_mu, self.b_prior_mu)
            prior_sig_t = propup_softplus(prior_4_t, self.W_prior_sig, self.b_prior_sig)

            # Inference term
            enc_1_t = propup_relu([o_t, p_t, h_tm1], self.W_enc1, self.b_enc1)
            enc_2_t = propup_relu(enc_1_t, self.W_enc2, self.b_enc2)
            enc_3_t = propup_relu(enc_2_t, self.W_enc3, self.b_enc3)
            enc_4_t = propup_relu(enc_3_t, self.W_enc4, self.b_enc4)
            enc_mu_t = propup_linear(enc_4_t, self.W_enc_mu, self.b_enc_mu)
            enc_sig_t = propup_softplus(enc_4_t, self.W_enc_sig, self.b_enc_sig)

            z_t = self.Gaussian_sample(enc_mu_t, enc_sig_t)

            # Compute Z network
            z_1_t = propup_relu(z_t, self.W_z1, self.b_1)
            z_2_t = propup_relu(z_1_t, self.W_z2, self.b_2)
            z_3_t = propup_relu(z_2_t, self.W_z3, self.b_3)
            z_4_t = propup_relu(z_3_t, self.W_z4, self.b_4)

            # Compute new recurrent hidden state
            h_t, c_t = self.lstm_prop([o_t, z_4_t], c_tm1, h_tm1)

            return h_t, c_t, enc_mu_t, enc_sig_t, prior_mu_t, prior_sig_t, z_4_t

        # Scan through input sequence
        ((h, c, enc_mu, enc_sig, prior_mu, prior_sig, z_4), updates) =\
            theano.scan(fn=inner_fn,
                        sequences=[o_4, p_4],
                        outputs_info=[h_0, c_0, None, None, None, None, None])

        # Reconstruction from inferred z_t
        # Can be performed after scanning, which is computationally more efficient
        dec_1 = propup_relu([z_4, p_4, h], self.W_dec1, self.b_dec1)
        dec_2 = propup_relu(dec_1, self.W_dec2, self.b_dec2)
        dec_3 = propup_relu(dec_2, self.W_dec3, self.b_dec3)
        dec_4 = propup_relu(dec_3, self.W_dec4, self.b_dec4)
        dec_mu = propup_linear(dec_4, self.W_dec_mu, self.b_dec_mu)
        dec_sig = propup_softplus(dec_4, self.W_dec_sig, self.b_dec_sig)

        # We need :
        #   - prior : p(z_t) (w/ reparametrization trick, just pass mu and sigma)
        #   - approx inference : q(z_t|x_t)
        #   - reconstruction : p(x|z)
        return enc_mu, enc_sig, prior_mu, prior_sig, dec_mu, dec_sig, updates

    def compute_nll_upper_bound(self, validation=False):
        #############
        # Inference
        enc_mu, enc_sig, prior_mu, prior_sig, dec_mu, dec_sig, updates_inference = \
            self.inference(self.orch, self.piano)

        #############
        # Cost
        recon = Gaussian(self.orch, dec_mu, dec_sig)
        kl = KLGaussianGaussian(enc_mu, enc_sig, prior_mu, prior_sig)
        # Mean over batches
        recon_term = recon.mean()
        kl_term = kl.mean()
        # neg log-lik upper bound
        cost = recon_term + kl_term

        #############
        # Gradient
        gparams = T.grad(cost, self.params)

        #############
        # Updates
        updates_train = self.optimizer.get_updates(self.params, gparams)

        #############
        # Monitor training
        # We use, as in Bengio a dictionary
        monitor = OrderedDict()
        # does the upper bound decrease ?
        monitor['nll_upper_bound'] = cost

        if validation:
            # If validation, compute more monitoring values
            monitor['recon_term'] = recon_term
            monitor['kl_term'] = monitor

            # Original values
            max_orch = self.orch.max()
            mean_orch = self.orch.mean()
            min_orch = self.orch.min()
            monitor['max_orch'] = max_orch
            monitor['mean_orch'] = mean_orch
            monitor['min_orch'] = min_orch

            # Reconstructed distribution
            max_recon_orch_mu = dec_mu.max()
            mean_recon_orch_mu = dec_mu.mean()
            min_recon_orch_mu = dec_mu.min()
            monitor['max_recon_orch_mu'] = max_recon_orch_mu
            monitor['mean_recon_orch_mu'] = mean_recon_orch_mu
            monitor['min_recon_orch_mu'] = min_recon_orch_mu

            max_recon_orch_mu = dec_mu.max()
            mean_recon_orch_mu = dec_mu.mean()
            min_recon_orch_mu = dec_mu.min()
            monitor['max_recon_orch_mu'] = max_recon_orch_mu
            monitor['mean_recon_orch_mu'] = mean_recon_orch_mu
            monitor['min_recon_orch_mu'] = min_recon_orch_mu

        # Cost is in monitor
        return monitor, updates_train

    def cost_update(self):
        return self.compute_nll_upper_bound(validation=False)

    def validation(self):
        # Validation = computing the nll upper bound on the validation set
        # So this function simply is a wrapper of cost_update function w/ no updates
        monitor, _ = self.compute_nll_upper_bound(validation=True)
        # observe the separate evolution of the recognition & approximation terms
        monitor['recon'] = recon_term
        monitor['kl_term'] = kl_term
        # observe the reconstruction x_tilde and latent z
        stats = enc_mu, enc_sig, prior_mu, prior_sig, dec_mu, dec_sig
        monitor['stats'] = stats
        #   - max mu_z, sigma_z
        return monitor

    # def generate(self):
