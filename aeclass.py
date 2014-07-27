import numpy as np
import theano
import theano.tensor as T
from theano import shared
from theano import config
from theano.tensor.shared_randomstreams import RandomStreams 

class Autoencoder(object):
    def __init__(self, np_rng, input=None,  n_vis=784, n_hid=500, W=None, hbias = None, vbias=None):
        # theano.config.floatX so that the code is runable on GPU
        self.n_visible = n_vis
        self.n_hidden = n_hid
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
        '''if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))
        self.theano_rng = theano_rng'''

    def get_hidden_values(self, input):
        """ computes values of hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b_hid)

    def get_reconstructed_input(self, hidden):
        """ computes the reconstructed input given the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_vis)
    '''
    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    '''

    def get_cost_updates(self, learning_rate):
        """ computes the cost and the updates for one trainng step """
        y = self.get_hidden_values(self.x)
        #tilde_x = self.get_corrupted_input(self.x, corruption_level)
        #y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : we sum over the size of a datapoint; if we are using minibatches, L will  be a vector, with one entry per example in minibatch.
        # We need to compute the average of all these to get the cost of the minibatch
        #print L
        cost = T.mean(L)
        gparams = T.grad(cost, self.params) # compute the gradients of the cost of the `dA` with respect to its parameters
        updates = [] #generate list of updates
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        return (cost, updates)
