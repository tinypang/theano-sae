#import cPickle
#import gzip
#import os
#import sys

from format_dataset import split_dataset,relabel_dataset
from import_dataset import import_dataset
from logistic_sgd import load_data
import numpy as np

def count(labels):
    d = {}
    for i in labels:
        if i in d:
            d[i] +=1
        else:
            d[i] = 1
    return d

def check(dataset):
    x,y = dataset
    counted = count(y)
    print counted
    return counted

def stats(dictionary):
    counts = np.array(dictionary.values())
    print 'mean'
    mean =  np.mean(counts)
    print mean
    print 'std dev'
    std = np.std(counts)
    print std
    print 'standardised counts'
    stdcounts = np.subtract(counts,mean)
    stdcounts = np.ceil(np.divide(stdcounts,std))
    print stdcounts

def process(dataset):
    print 'train'
    train = check(dataset[0])
    stats(train)
    print 'valid'
    valid = check(dataset[1])
    stats(valid)
    print 'test'
    test = check(dataset[2])
    stats(test)

data, labels = import_dataset('./spectrogram/ISMIR_genre/ismirg_3sec_50x20_gs/',input_type='spec',dataset='ismir',dimx=50,dimy=20,minmax=True)
dataset, label_dict = relabel_dataset(labels,data)
print label_dict
'''
print 'relabelling data'
relabels, labeldict = relabel_data(labels, 'classifier_label_mapping.txt')

print 'casting labels to arrays'
relabels = np.array(relabels)
labels = np.array(labels)


print 'splitting and counting labels on original label set'
dataset = split_dataset(data, labels, 28,28)
print 'original labels'
process(dataset)
'''
n = 1
for i in range(0,n):
    print i
    print 'splitting and counting labels on relabelled set'
    dataset_relabelled = split_dataset(dataset,8)
    print 'relabelled'
    process(dataset_relabelled)
    print '--------------------------------------------------------------------------------'

