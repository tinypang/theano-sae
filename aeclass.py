import numpy as np
import theano
import theano.tensor as T
from theano import shared
from theano import config

class Autoencoder(object):
    def __init__(self, np_rng, input=None, n_vis=784, n_hid=500, W=None, hbias = None, vbias=None):
        # theano.config.floatX so that the code is runable on GPU
        
        if not W:
            init_W = np.asarray(np_rng.uniform(low=-4 * np.sqrt(6. / (n_hid + n_vis)),high=4 * np.sqrt(6. / (n_hid + n_vis)),size=(n_vis, n_hid)), dtype=config.floatX)
            W = shared(value=init_W, name = 'W')
        if not vbias:
            vbias = shared(value=np.zeros(n_vis, dtype=config.floatX), name='vbias')
        if not hbias:
            hbias = shared(value=np.zeros(n_hid, dtype=config.floatX), name='hbias')
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
        y = self.get_hidden_values(self.x)
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
