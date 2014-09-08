import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, shared_dataset
from mlp import HiddenLayer
from dA import dA
from stacked_ae import SdA
from pca_whiten import pca
from utils import tile_raster_images
from format_dataset import split_dataset
from import_mpc import import_mpc

'''
def pr_mat(conf_matrix):    #todo integrate label dict to use labels instead of int ids
    pr_mat = {}
    keys_col = conf_matrix.keys()
    for i in conf_matrix.keys():
        pr_mat[i] = {}
        #calculate recall
        rowsum = 0
        for j in conf_matrix[i].keys():
            rowsum+= conf_matrix[i][j]
        pr_mat[i]['recall'] = conf_matrix[i][i]/rowsum
        #calculate precision
        colsum = 0
        for k in keys_col:
            colsum += conf_matrix[k][i]
        pr_mat[i]['precision']= conf_matrix[i][i]/colsum
    return pr_mat    
'''    

def test_SdA(path='spectrogram/preprocessed_50th_full',finetune_lr=0.005, pretraining_epochs=125,
             pretrain_lr=0.0005, training_epochs=2000,
            batch_size=1,dimx=2,dimy=297,hidlay=[300,300],outs=100,corruption_levels=[.35, .35],input_type='mpc',nceps=33,valid_imp_thresh=0.5,resultslog='resultslog.txt',reinput=None):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset
    
    :type input_type: string
    :param dataset: should be 'mpc' or 'spectrogram' determines input type

    """

    log = open(resultslog,'a+')
    log.write('data set is {0}\n'.format(path))
    log.write("Xdim:{0}, Ydim:{1}, Hidden Layers:{2}, nOutputs:{3}, Batch size:{4}, pretraining epochs:{5}, pretrain learning rate:{6}, finetuning learning rate:{7}, training epochs:{8}, corruption levels per layer:{9}, validation improvement threshold:{10}\n".format(dimx,dimy,str(hidlay),outs,batch_size,pretraining_epochs,pretrain_lr,finetune_lr,training_epochs,corruption_levels,valid_imp_thresh))
    log.write('input type: {0}'.format(input_type))
    if reinput != None:
        datasets = reinput    
    else:    
        if input_type == 'spectrogram':
            pcaonoff = False
            pcancomp = 0
            log.write('pca:{0}, pca components:{1}\n'.format(pcaonoff,pcancomp))
            data, labels = pca(path,dimx=dimx,dimy=dimy,ncomp=pcancomp,whiten=pcaonoff)
        if input_type == 'mpc':
            scale = True
            whiten = False
            pcancomps = 100
            log.write('number of mpc coefficients:{0},scale: {1}, whiten: {2}, ncomps{3}'.format(nceps,scale,whiten, pcancomps))
            data, labels =  import_mpc(path,nceps,scale=scale,whiten=whiten,ncomps=pcancomps)
        datasets = split_dataset(data, labels)
    label_dict = datasets[3]
    pairs = zip(label_dict.itervalues(), label_dict.iterkeys())
    pairs.sort()
    for i in pairs:
        log.write(str(i))
    log.write('\n')
    n_classes = len(label_dict.keys())
    train_set_x, train_set_y = shared_dataset(datasets[0])
    #valid_set_x, valid_set_y = shared_dataset(datasets[1])
    #test_set_x, test_set_y = shared_dataset(datasets[2])
    data, labels = [], []    
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(numpy_rng=numpy_rng, n_ins=dimx*dimy,
              hidden_layers_sizes=hidlay,
              n_outs=outs,n_classes=n_classes,corruption_levels=corruption_levels)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    t1 = time.clock()
    ## Pre-train layer-wise
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
            t2 = time.clock()
            print 'epoch took {0} sec'.format(t2-t1)
            t1 = t2

    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    log.write('The pretraining code for file {0} ran for {1:.2f}m\n'.format(os.path.split(__file__)[1],(end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################
    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model, conf_mat = sda.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)

    print '... finetunning the model'
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = valid_imp_thresh  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    confusion_matrix = T.imatrix
    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        #test_pr[epoch] = {}
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    print 'new best validation model found'
                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    print 'saving best validation score'
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    # test it on the test set
                    print 'testing best model on test data'
                    test_losses = test_model()
                    confusion_matrix = conf_mat()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    log.write(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.)+'\n')

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print confusion_matrix
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    log.write(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.)+'\n')
    log.write('The training code for file {0} ran for {1:.2f}\n'.format(os.path.split(__file__)[1],((end_time - start_time) / 60.)))
    log.write(str(confusion_matrix)+'\n')
    log.write('----------\n')
    log.close()
    return datasets

if __name__ == '__main__':
    #test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20)
    #test_SdA(path='spectrogram/test',batch_size=20)
    test_SdA(path='./GTZAN_genre',batch_size=20,input_type='mpc',nceps=33)

