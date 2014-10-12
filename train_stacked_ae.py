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
from import_dataset import import_dataset
from utils import tile_raster_images
from format_dataset import relabel_dataset,split_dataset
import PIL.Image 
import re

def test_SdA(path='spectrogram/preprocessed_50th_full',nfolds=1,finetune_lr=0.005, pretraining_epochs=125,
             pretrain_lr=0.0005, training_epochs=2000,
            batch_size=1,dimx=2,dimy=297,hidlay=[300,300],outs=100,corruption_levels=[.35, .35],input_type='mpc',nceps=33,valid_imp_thresh=0.995,resultslog='resultslog.txt',reinput=None,shuffle=False, patience_m=10,patience_increase_m = 2.):
    """
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
    log.write("Xdim:{0}, Ydim:{1}, Hidden Layers:{2}, nOutputs:{3}, Batch size:{4}, pretraining epochs:{5}, pretrain learning rate:{6}, finetuning learning rate:{7}, training epochs:{8}, corruption levels per layer:{9}, validation improvement threshold:{10},dataset type:{11},nfolds: {12}\n".format(dimx,dimy,str(hidlay),outs,batch_size,pretraining_epochs,pretrain_lr,finetune_lr,training_epochs,corruption_levels,valid_imp_thresh,input_type,nfolds))
    log.write('input type: {0}'.format(input_type))
    
    ismir = re.compile('.*ismir.*') #identify dataset for label determination purposes
    gtzan = re.compile('.*GTZAN.*')
    if ismir.match(path) != None:
        dataset_name = 'ismir' 
    elif gtzan.match(path) != None:
        dataset_name = 'gtzan'
    else:
        print 'unrecognised dataset type'
        sys.exit()

    pca = False
    whiten = False
    pcancomps = 0
    minmax = True
    log.write('pca:{0}, pca components:{1}\n'.format(pca,pcancomps))
    if input_type == 'spec':
        n_ins = dimx*dimy
        log.write('minmaxscale: {0}, whiten: {1}, ncomps{2}'.format(minmax,whiten, pcancomps))
    elif input_type == 'mpc':
        n_ins = (nceps*(nceps+1))/2 + nceps
        log.write('number of mpc coefficients:{0}, minmaxscale: {1}, whiten: {2}, ncomps{3}'.format(nceps,minmax,whiten, pcancomps))
    if pca == True:
        n_ins = pcancomps
        log.write('pca components: {0}\n'.format(pcancomps))
    if reinput != None:
        dataset,label_dict = reinput
    #if (extra_test != None) & (isinstance(extra_test,bool)==False):
    #    test_dataset = extra_test
    else:
        data, labels = import_dataset(path,input_type=input_type,dataset=dataset_name,dimx=dimx,dimy=dimy,nceps=33,ncomp=pcancomps,pca=pca,whiten=whiten,minmax=minmax)
        dataset,label_dict =  relabel_dataset(labels,data)   #relabel data
        '''
        if (dataset_name == 'ismir')&(extra_test == True):
            test = import_dataset('./audio_snippets/ismir_dev_test',input_type=input_type,dataset=dataset_name,dimx=dimx,dimy=dimy,nceps=33,ncomp=pcancomps,pca=pca,whiten=whiten,minmax=minmax)
            test_dataset,label_dict2 = relabel_dataset(labels,data,label_dict)
        '''

    fold_scores = []   #initiate list of fold scores
    n_classes = len(label_dict.keys())

    pairs = zip(label_dict.itervalues(), label_dict.iterkeys()) #write to file label mappings
    pairs.sort()
    for i in pairs:
        log.write(str(i))
    log.write('\n')
    '''
    if label_dict2:
        log.write('test data mapping')
        pairs = zip(label_dict2.itervalues(), label_dict2.iterkeys()) #write to file test label mappings
        pairs.sort()
        for i in pairs:
            log.write(str(i))
        log.write('\n')
    '''
    for fold in range(0,nfolds):
        print 'testing fold {0}'.format(fold)
        log.write('fold: {0}\n'.format(fold))
        '''
        if (dataset_name == 'ismir')&(test_dataset!=None):
            datasets = split_dataset(dataset,n_classes,shuffle,test_dataset)
        else:    
            datasets = split_dataset(dataset,n_classes,shuffle)
        '''
        datasets = split_dataset(dataset,n_classes,shuffle)
        train_set_x, train_set_y = shared_dataset(datasets[0])
        data, labels = [], []    
        
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size

        # numpy random generator
        numpy_rng = numpy.random.RandomState(89677)
        print '... building the model'
        # construct the stacked denoising autoencoder class
        sda = SdA(numpy_rng=numpy_rng, n_ins=n_ins,n_classes=n_classes,
                  hidden_layers_sizes=hidlay,
                  n_outs=outs)

        #########################
        # PRETRAINING THE MODEL #
        #########################
        print '... getting the pretraining functions'
        pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size)

        print '... pre-training the model'
        start_time = time.time()
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

        end_time = time.time()

        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

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
        patience = patience_m * n_train_batches
        patience_increase = patience_increase_m
        #patience = 10 * n_train_batches  # look as this many examples regardless
        #patience_increase = 2.  # wait this much longer when a new best is
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
        start_time = time.time()

        done_looping = False
        epoch = 0

        while (epoch < training_epochs) and (not done_looping):
            epoch = epoch + 1
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

                        #improve patience if loss improvement is good enough
                        if (this_validation_loss < best_validation_loss *
                            improvement_threshold):
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = test_model()
                        test_score = numpy.mean(test_losses)
                        confusion_matrix = conf_mat()
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    log.write('stopped at epoch {0}\n'.format(epoch))
                    break


        end_time = time.time()
        print confusion_matrix
        print(('Optimization complete with best validation error of %f %%,'
               'with test performance %f %%') %
                     (best_validation_loss * 100., test_score * 100.))
        print >> sys.stderr, ('The training code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        log.write(('Optimization complete with best validation error of %f %%,'
               'with test performance %f %%') %
                     (best_validation_loss * 100., test_score * 100.)+'\n')
        fold_scores.append([100- (best_validation_loss * 100.),100-(test_score * 100.)])
        log.write('The training code for file {0} ran for {1:.2f}m for fold {2}\n'.format(os.path.split(__file__)[1],((end_time - start_time) / 60.),fold))
        log.write(str(confusion_matrix)+'\n')
        
        for layer in range(0,len(sda.sigmoid_layers)):
            x = sda.sigmoid_layers[layer].W.get_value(borrow=True).T
            tile = x.shape[0]
            img = x.shape[1]
            if img == 594:
                dim = (18,33)
            elif img == 784:
                dim = (28,28)
            else:
                h = 20
                w = img/h
                dim = (h,w)
            image = PIL.Image.fromarray(tile_raster_images(X=x,img_shape=dim, tile_shape=(int(tile/10),10),tile_spacing=(1,1)))
            image.save('./feature_maps/{0:.0f}_{1}_{2}_{3}_fold_{4}_features_layer_{5}.png'.format(time.time(),dataset_name,input_type,dimx*dimy,fold,layer))
    
    print 'validation and test accuracy across all folds was'
    print fold_scores
    log.write('validation and test accuracy across all folds was\n')
    log.write(str(fold_scores)+'\n')
    fold_scores = numpy.array(fold_scores)
    avg_score = numpy.mean(fold_scores,axis=0)
    print 'best validation accuracy was {0} and best test accuracy was {1}'.format(avg_score[0], avg_score[1])
    log.write('best validation accuracy was {0} and best test accuracy was {1}\n'.format(avg_score[0], avg_score[1])) 
    log.write('----------\n')
    log.close()
    '''
    if (dataset_name == 'ismir')&(test_dataset != None):
        return (dataset,label_dict),test_dataset
    else:
        return (dataset, label_dict)
    '''
    return (dataset, label_dict)

if __name__ == '__main__':
    test_SdA(path='./audio_snippets/ismir_test',nfolds=10,input_type='mpc',dimx=33,dimy=18,batch_size=20,hidlay=[400,200,100],outs=50,corruption_levels=[.2, .2, .2],nceps=33,finetune_lr=0.1, pretraining_epochs=25,pretrain_lr=0.001,training_epochs=2000,shuffle = True)
    #test_SdA(path='./audio_snippets/GTZAN_3sec',input_type='mpc',dimx=33,dimy=18,batch_size=20,hidlay=[400,200,100],outs=50,corruption_levels=[.0, .0, .0],nceps=33,finetune_lr=0.1, pretraining_epochs=100,pretrain_lr=0.001)
