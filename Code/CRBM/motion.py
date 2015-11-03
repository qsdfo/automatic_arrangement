""" Mocap data
See: http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/code.html

Download:
http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/motion.mat
Place in ../data

Data originally from Eugene Hsu, MIT.
http://people.csail.mit.edu/ehsu/work/sig05stf/

@author Graham Taylor
"""
import scipy.io
import numpy as np
from numpy import arange
import theano

def preprocess_data(Motion):

    n_seq = Motion.shape[1]

    # assume data is MIT format for now
    indx = np.r_[
        arange(0,6),
        arange(6,9),
        13,
        arange(18,21),
        25,
        arange(30,33),
        37,
        arange(42,45),
        49,
        arange(54,57),
        arange(60,63),
        arange(66,69),
        arange(72,75),
        arange(78,81),
        arange(84,87),
        arange(90,93),
        arange(96,99),
        arange(102,105)]

    row1 = Motion[0,0][0]

    offsets =   np.r_[
        row1[None,9:12],
        row1[None,15:18],
        row1[None,21:24],
        row1[None,27:30],
        row1[None,33:36],
        row1[None,39:42],
        row1[None,45:48],
        row1[None,51:54],
        row1[None,57:60],
        row1[None,63:66],
        row1[None,69:72],
        row1[None,75:78],
        row1[None,81:84],
        row1[None,87:90],
        row1[None,93:96],
        row1[None,99:102],
        row1[None,105:108]]

    # collapse sequences
    batchdata = np.concatenate([m[:, indx] for m in Motion.flat], axis=0)

    data_mean = batchdata.mean(axis=0)
    data_std = batchdata.std(axis=0)

    # Normalization
    batchdata = (batchdata - data_mean) / data_std

    # get sequence lengths
    seqlen = [s.shape[0] for s in Motion.flat]


    return batchdata, seqlen, data_mean, data_std

def load_data(filename):

    # load data post preprocess1
    mat_dict = scipy.io.loadmat(filename)
    Motion = mat_dict['Motion']

    batchdata, seqlen, data_mean, data_std = preprocess_data(Motion)

    # put data into shared memory
    shared_x = theano.shared(np.asarray(batchdata, dtype=theano.config.floatX))

    return shared_x, seqlen, data_mean, data_std

if __name__ == "__main__":
    batchdata, seqlen, data_mean, data_std = load_data('../data/motion.mat')
