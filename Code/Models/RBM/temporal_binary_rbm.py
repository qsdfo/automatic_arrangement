"""
A temporal RBM with binary visible units.
"""
import numpy

import theano
import theano.tensor as T

import Score_function

from theano.tensor.shared_randomstreams import RandomStreams

from Data_processing.load_data import load_data


class RBM_temporal_bin(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        input=None,
        past=None,
        n_hidden=500,
        W=None,
        P=None,
        hbias=None,
        vbias=None,
        pbias=None,
        numpy_rng=None,
        theano_rng=None
    ):

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')
        self.past = past
        if not past:
            self.past = T.matrix('past')

        # Architecture
        self.n_hidden = n_hidden
        self.n_visible = self.input.get_value(borrow=True).shape[0]
        self.n_past = self.past.get_value(borrow=True).shape[0]

        # Initialize random generators
        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + (self.n_visible + self.n_past))),
                    high=4 * numpy.sqrt(6. / (n_hidden + (self.n_visible + self.n_past))),
                    size=(self.n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if P is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_P = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + (self.n_visible + self.n_past))),
                    high=4 * numpy.sqrt(6. / (n_hidden + (self.n_visible + self.n_past))),
                    size=(self.n_past, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            P = theano.shared(value=initial_P, name='P', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    self.n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        if pbias is None:
            # create shared variable for visible units bias
            pbias = theano.shared(
                value=numpy.zeros(
                    self.n_past,
                    dtype=theano.config.floatX
                ),
                name='pbias',
                borrow=True
            )

        self.W = W
        self.P = P
        self.hbias = hbias
        self.vbias = vbias
        self.pbias = pbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.P, self.hbias, self.vbias, self.pbias]

    def free_energy(self, v, p):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v, self.W) + T.dot(p, self.P) + self.hbias
        vbias_term = T.dot(v, self.vbias) + T.dot(p, self.pbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def gibbs_step(self, v, p):
        mean_h = T.nnet.sigmoid(T.dot(v, self.W) + T.dot(p, self.P) + self.hbias)
        h = self.theano_rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                                     dtype=theano.config.floatX)
        mean_p = T.nnet.sigmoid(T.dot(h, self.P.T) + self.pbias)
        mean_v = T.nnet.sigmoid(T.dot(h, self.W.T) + self.vbias)
        v = self.theano_rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                                     dtype=theano.config.floatX)
        p = self.theano_rng.binomial(size=mean_p.shape, n=1, p=mean_p,
                                     dtype=theano.config.floatX)
        return v, mean_v, p

    # Get cost and updates for training
    def cost_updates(self, lr=0.1, k=1):
        # Negative phase
        visible_chain, mean_visible_chain, past_chain, updates = theano.scan(self.gibbs_step,
                                                                             outputs_info=[self.input, None, self.past],
                                                                             n_steps=k)
        neg_v = visible_chain[-1]
        mean_neg_v = mean_visible_chain[-1]
        neg_p = past_chain[-1]

        # Cost
        cost = T.mean(self.free_energy(self.input, self.past)) -\
            T.mean(self.free_energy(neg_v, neg_p))

        # Gradient
        gparams = T.grad(cost, self.params, consider_constant=[neg_v, neg_p])

        # Updates
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )

        # Monitor reconstruction (log-likelihood proxy)
        monitoring_cost = self.get_reconstruction_cost(updates, mean_neg_v)

        return monitoring_cost, updates

    def get_reconstruction_cost(self, nv):
        """Approximation to the reconstruction error """
        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(nv) +
                (1 - self.input) * T.log(1 - nv),
                axis=1
            )
        )
        return cross_entropy

    #  Over-fitting monitoring
    def overfit_measure(self, v_validate, p_validate):
        free_en_diff = self.free_energy(v_validate, p_validate) - self.free_energy(self.input, self.past)
        return free_en_diff / self.free_energy(v_validate, p_validate)

    # Sampling with clamped past units
    # Two methods :
    #   - by alternate Gibbs sampling
    def sampling_Gibbs(self, k=20):
        # Negative phase with clamped past units
        visible_chain, mean_visible_chain, past_chain, updates = theano.scan(self.gibbs_step,
                                                                             outputs_info=[self.input],
                                                                             non_sequences=[None, self.past],
                                                                             n_steps=k)

        pred_v = visible_chain[-1]
        mean_pred_v = mean_visible_chain[-1]
        return pred_v, mean_pred_v

    def prediction_measure(self, k=20):
        pred_v, mean_pred_v = self.sampling_Gibbs(k)
        precision = Score_function.prediction_measure(self.input, mean_pred_v)
        recall = Score_function.recall_measure(self.input, mean_pred_v)
        accuracy = Score_function.accuracy_measure(self.input, mean_pred_v)

        return precision, recall, accuracy


def train(hyper_parameter, dataset, output_folder, output_file):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    """
    # Load parameters
    n_hidden = hyper_parameter['n_hidden']
    temporal_order = hyper_parameter['temporal_order']
    learning_rate = hyper_parameter['learning_rate']
    training_epochs = hyper_parameter['training_epochs']
    batch_size = hyper_parameter['batch_size']

    # First check if this configuration has not been tested before,
    # i.e. its parameter are written in the result.csv file
    dataset, train_index, validate_index, test_index = load_data(dataset, temporal_order, batch_size, False, (0.7, 0.1, 0.2))

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_index.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    v = T.matrix('v')  # the data is presented as rasterized images
    p = T.matrix('p')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # construct the RBM class
    rbm = RBM_temporal_bin(input=v,
                           past=p,
                           n_hidden=n_hidden,
                           numpy_rng=rng,
                           theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.cost_updates(lr=learning_rate, k=1)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function([index, index_hist], cost,
                                 updates=updates,
                                 givens={x: batchdata[index],
                                 x_history: batchdata[index_hist].reshape((
                                         batch_size, delay * n_dim))},
                                 name='train_crbm')

    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            v: dataset[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))
    # end-snippet-5 start-snippet-6
    #################################
    #     Sampling from the RBM     #
    #################################
    # find out the number of test samples
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        numpy.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )
    # end-snippet-6 start-snippet-7
    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'

    )
    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = numpy.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype='uint8'
    )
    for idx in xrange(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print ' ... plotting sample ', idx
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save('samples.png')
    # end-snippet-7
    os.chdir('../')

if __name__ == '__main__':
    # Hyper-parameter
    hyper_parameter = {}
    hyper_parameter['n_hidden'] = 500
    hyper_parameter['temporal_order'] = 10
    hyper_parameter['learning_rate'] = 0.1
    hyper_parameter['training_epochs'] = 1000
    hyper_parameter['batch_size'] = 100,
    # File
    dataset = '../../../Data/data.p',
    output_folder = 'rbm_plots'
