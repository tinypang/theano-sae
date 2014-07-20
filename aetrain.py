import cPickle
import gzip
import os
import sys
import theano
import theano.tensor as T
import numpy as np
import aeclass as ae
import time
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from logistic_sgd import load_data
from utils import tile_raster_images

import PIL.Image
'''
# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')

test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)

batch_size = 500    # size of the minibatch

# accessing the third minibatch of the training set

data  = train_set_x[2 * 500: 3 * 500]
label = train_set_y[2 * 500: 3 * 500]
'''

def test_dA(learning_rate=0.1, training_epochs=15,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='dA_plots'):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    #get dataset
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    #build model
    rng = np.random.RandomState(123)
    #theano_rng = RandomStreams(rng.randint(2 ** 30))
    autoencoder = ae.Autoencoder(np_rng=rng, n_vis=784, n_hid=500)
    cost, updates = autoencoder.get_cost_updates(learning_rate=0.1)
    train_ae = function([index], outputs=cost, updates=updates, givens={x: train_set_x[index * batch_size:(index + 1) * batch_size]}, on_unused_input = 'warn', name='train_ae')
    #train = theano.function([x], cost, updates=updates, on_unused_input='warn') 
    start_time = time.clock ()

    #training

    for epoch in xrange(training_epochs):
        # go through training set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_ae(batch_index))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    end_time = time.clock

    training_time = (end_time - start_time)

    print ('Training took %f minutes' % (pretraining_time / 60.))
    image = PIL.Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')

if __name__ == '__main__':
    test_dA()
