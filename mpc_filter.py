import numpy as np
import glob
import os
from mfcc import mfcc
import sys
from scikits.audiolab import auread, wavread
from scikits.audiolab import Format, Sndfile
import math
import matplotlib.pyplot as plt
import sunau
import re
import sys

n = 0
data = []
labels = []

def explore(path,ncep,dataset):
    global n
    if dataset == 'gtzan':
        sounds = glob.glob(path + '/*.au')  #find all files with .au file extension
    elif dataset == 'ismir':
        sounds = glob.glob(path + '/*.wav')  #find all files with .wave file extension
    else:
        print 'unrecognised dataset type'
        sys.exit()
    for filename in os.listdir(path):   #for all files in the given directory
        if sounds.count(path + '/' + filename) != 0:    #if the file ends in .au, graph it's spectrogram
            get_mpc(path + '/' + filename,filename,ncep,dataset)            
            n+=1
            print n
        else:                       #if the file is a folder, recursively explore it
            explore(path + '/' + filename,ncep,dataset)

def get_mpc(path,filename,ncep,dataset):
    if dataset == 'gtzan':
        signal, rate, enc = auread(path)
    elif dataset == 'ismir':
        signal, rate, enc = wavread(path)
    else:
        print 'unrecognised dataset type'
    mfcc_feat,mpc_feat,spectrum  = mfcc(signal,fs=rate,nfft=512,nceps=ncep,nwin=512)
    feat = mpc_feat[:,:ncep]
    mean_feat = np.mean(feat, axis=0)
    covvar = np.cov(feat,rowvar=0)
    cov_feat = []
    for i in range(0,covvar.shape[0]):
        for j in range(0,i+1):
            cov_feat.append(covvar[i][j])
    mean_feat = mean_feat.tolist()
    feature_vector = []
    feature_vector.extend(mean_feat)
    feature_vector.extend(cov_feat)
    feature_vector = np.array(feature_vector)
    data.append(feature_vector)
    if dataset == 'ismir':
        labels.append(re.split('-', filename, maxsplit=1)[0])
    elif dataset == 'gtzan':
        labels.append(re.split('\.', filename, maxsplit=1)[0])
    else:
        print 'unrecognised dataset type'
        sys.exit()
    
def mpcfilter(path,dataset,ncep=33):
    explore(path,ncep,dataset)
    return data, labels

if __name__ == '__main__':
    mpcfilter('./test',33)
    




