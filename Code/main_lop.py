import numpy

def split_data(dataset,indices):
    print '... splitting between train, validate and test data'

    train_dataset = dataset[indices[0]]
    validate_dataset = dataset[indices[1]]
    test_dataset = dataset[indices[2]]

    print '... sharing data'
    # The structure loaded contains :
    #       - the dataset represented as a matrix
    #       - train, validate and test indices which is a dictionnary of matrices
    # Why thius structure ? Because since our DB is small, we test over the wholle
    # DB by permuting the test indices by batch of 10%
    #
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

    test_set_piano, test_set_orchestra = shared_dataset(test_set)
    valid_set_piano, valid_set_orchestra = shared_dataset(valid_set)
    train_set_piano, train_set_orchestra = shared_dataset(train_set)

    rval = [(train_set_piano, train_set_orchestra), (valid_set_piano, valid_set_orchestra),
            (test_set_piano, test_set_orchestra)]
    return rval

# ################################################################################
# ################################################################################
# ###############             Training functions
def train_model(
    # Dataset parameters
        dataset,
        indices,
        batch_size,
    # Model parameters
        model_name,
        layers,
        numfac=None,
        numfeat=None,
        temporal_order, # In beat. So absolute temporal_order depends on the quantization used when building the db
    # Training parameters
        num_epochs=2000,    # Should not be reached
        CD_K=1,
        learning_rate=10.^-3,
        weight_decay=None, # Default is no weight decay...
        sparsTarget=None,
        sparsBeta=None,
        sparsLambda=None,
        momentum=None
):



    train_set_orchestra, valid_set_orchestra, test_set_orchestra = split_dataset(full_dataset_orchestra, shuffeling, index_train)

    train_set_piano, train_set_orchestra = datasets[0]
    valid_set_piano, valid_set_orchestra = datasets[1]
    # test_set_piano, test_set_orchestra = datasets[2]
    # Test sera une autre fonction

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')    # input matrix, concatenation

    rng = numpy.random.RandomState(1234)

########################################################################
########################################################################
######## Model Instanciation
    print '... building the model'

    if model_name=='RBM':
        # construct the RBM class
        network = RBM(input=x, n_visible=28 * 28,
                  n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

        # get the cost and the gradient corresponding to one step of CD-15
        cost, updates = network.get_cost_updates(lr=learning_rate,
                                             persistent=persistent_chain, k=CD_K)

        # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch
        test_model = theano.function(
            inputs=[index],
            outputs=network.errors(x),  # A écrire
            givens={
                x: concatenate(
                    (test_set_piano[index * batch_size: (index + 1) * batch_size],test_set_orchestra[index * batch_size: (index + 1) * batch_size],), axis=0),
            }
        )

        validate_model = theano.function(
            inputs=[index],
            outputs=network.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    elif:
########################################################################
########################################################################


########################################################################
########################################################################
######## Training
    print '... training'

    # Call training function
    # All the models share the same training/validating mechanism
    # early-stopping parameters
    patience = 10000  # look at this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    #####################
    #####################
    # Training is useless for some models (repeat, random...)
    if(model_name == 'Repeat' or model_name == 'Random'):
        num_epochs = 0
    #####################
    #####################

    while (epoch < num_epochs) and (not done_looping):
        epoch = epoch + 1
        # go through the training set
        mean_cost = []
        for minibatch_index in xrange(n_train_batches):

            # Call training function
            mean_cost += [train_model(batch_index)]

            # Validation mechanism
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    # Return the trained network which is a python object
    return network
########################################################################
########################################################################

if __name__ == '__main__':

    # Load data
    dataset, indices = load()

    # Loop over the strcuture of indices

    for iteration in xrange(1,numpy.shape(indices)[2]):

        local_indices = indices[,,iteration]

        # Train the network
        network = train_model(
            # Dataset parameters
                dataset = dataset,
                indices = local_indices,
                batch_size=100,
            # Model parameters
                model_name='RBM',
                layers=500,
                numfac=None,
                numfeat=None,
                temporal_order=None, # In beat. So it depends on the rhythm
            # Training parameters
                num_epochs=100,    # Should not be reached
                CD_K=10,
                learning_rate=10.^-3,
                weight_decay=None, # Default is no weight decay...
                sparsTarget=None,
                sparsBeta=None,
                sparsLambda=None,
                momentum=None
        )

        # Save the network
        pickle(network)
          ou
        numpy.save(network)  ????

        # Evaluation (predictive task)
