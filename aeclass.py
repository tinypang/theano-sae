import numpy as np
import theano as T
from theano import shared as shared
from theano import config

class Autoencoder(object):
    def __init__(self, np_rng, input=None, n_vis=784, n_hid=500,W=None, hbias = None, vbias=None):
		"""

    :type np_rng: numpy.random.RandomState
    :param np_rng: number random generator used to generate weights


    :type input: theano.tensor.TensorType
    :paran input: a symbolic description of the input or None for standalone
                  dA

    :type n_vis: int
    :param n_vis: number of visible units

    :type n_hidden: int
    :param n_hid:  number of hidden units

    :type W: theano.tensor.TensorType
    :param W: Theano variable pointing to a set of weights that should be
              shared belong the dA and another architecture; if dA should
              be standalone set this to None

    :type hbias: theano.tensor.TensorType
    :param hbias: Theano variable pointing to a set of biases values (for
                 hidden units) that should be shared belong dA and another
                 architecture; if dA should be standalone set this to None

    :type vbias: theano.tensor.TensorType
    :param vbias: Theano variable pointing to a set of biases values (for
                 visible units) that should be shared belong dA and another
                 architecture; if dA should be standalone set this to None


    """
        self.n_vis = n_vis
        self.n_hid = n_hid
        # W is initialized with `initial_W` which is uniformely sampled
        # from -4*sqrt(6./(n_visible+n_hidden)) and 4*sqrt(6./(n_hidden+n_visible))
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        if not W:
            init_W = np.asarray(np_rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_hid + n_vis)),
                high=4 * numpy.sqrt(6. / (n_hid + n_vis)),
                size=(n_vis, n_hid)), dtype=theano.config.floatX)
            W = shared(value=init_W, name = 'W')
        if not vbias:
            vbias = shared(value=np.zeroes(n_vis, dtype=theano.config.floatx), name='vbias')
        if not hbias:
            hbias = shared(value=np.zeroes(n_hid, dtype=theano.config.floatx), name='hbias')
        self.W = W
        self.b_hid = hbias
        self.b_vis = vbias
        self.W_prime = self.W.T     # tied weights, therefore W_prime is W transpose
        if input == None:           # if no input is given, generate a variable representing the input
            # we use a matrix because we expect a minibatch of several examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input          #use input as a parameter so we can stack layers of the autoencoder; output of kth layer is input of k+1th layer
        self.params = [self.W, self.b_hid, self.b_vis]

    def get_hidden_values(self, input):
    """ computes values of hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b_hid)

    def get_reconstructed_input(self, hidden):
    """ computes the reconstructed input given the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_vis)

    def get_cost_updates(self, learning_rate):
    """ computes the cost and the updates for one trainng step """
        y = self.get_hidden_values(self.x):
        z = self.get_reconstructed_input(y)
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : we sum over the size of a datapoint; if we are using minibatches, L will  be a vector, with one entry per example in minibatch.
        # We need to compute the average of all these to get the cost of the minibatch
        cost = T.mean(L)
        gparams = T.grad(cost, self.params) # compute the gradients of the cost of the `dA` with respect to its parameters
        updates = [] #generate list of updates
        for param, gparam in zip(self.params, gparams):
        updates.append((param, param - learning_rate * gparam))
        return (cost, updates)
