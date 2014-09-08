import numpy as np
import glob
import os
from mfcc import mfcc
import sys
from scikits.audiolab import auread
from scikits.audiolab import Format, Sndfile
import math
import matplotlib.pyplot as plt
import sunau


data = []
labels = []

def explore(path,ncep):
    sounds = glob.glob(path + '/*.au')  #find all files with .au file extension
    for filename in os.listdir(path):   #for all files in the given directory
        if sounds.count(path + '/' + filename) != 0:    #if the file ends in .au, graph it's spectrogram
            print filename
            get_mpc(path + '/' + filename,filename,ncep)            
        else:                       #if the file is a folder, recursively explore it
            explore(path + '/' + filename,ncep)

def get_audio_info(filepath):
    data, fs, enc = auread(filepath)
    return data, fs

def get_mpc(path,filename,ncep):
    signal, rate = get_audio_info(path)
    #ms = len(signal)/30000
    #winlen = 25*ms
    mfcc_feat,mpc_feat,spectrum  = mfcc(signal,fs=rate,nfft=512,nceps=ncep,nwin=512)
    feat = mpc_feat[:,:ncep]
    mean_feat = np.mean(feat, axis=0)
    covvar = np.cov(feat,rowvar=0)
    cov_feat = np.empty([0,0])
    for i in range(0,covvar.shape[0]):
        for j in range(0,i+1):
            cov_feat = np.append(cov_feat, covvar[i][j])            
    feature_vector = np.append(mean_feat,cov_feat)
    data.append(feature_vector)
    labels.append(filename[0:-11])
    
def mpcfilter(path,ncep=33):
    explore(path,ncep)
    return data, labels

if __name__ == '__main__':
    mpcfilter('./test',33)
    




