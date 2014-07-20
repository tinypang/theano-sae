import theano as T
import numpy as np
import aeclass as ae

autoencoder = Autoencoder(numpy_rng=numpy.random.RandomState(1234), n_vis=784, n_hid=500)
cost, updates = autoencoder.get_cost_updates(learning_rate=0.1)
train = theano.function([x], cost, updates=updates)
