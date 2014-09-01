import sunau
import numpy as np
import glob
import os
from scikits.talkbox.features.mfcc import mfcc as mfcc


data = []
labels = []

def explore(path,ncep):
    sounds = glob.glob(path + '/*.au')  #find all files with .au file extension
    for filename in os.listdir(path):   #for all files in the given directory
        if sounds.count(path + '/' + filename) != 0:    #if the file ends in .au, graph it's spectrogram
            get_mpc(path + '/' + filename,filename,ncep)            
            print filename
        else:                       #if the file is a folder, recursively explore it
            explore(path + '/' + filename,ncep)

def get_audio_info(filepath):
    wf = sunau.open(filepath, 'r')
    f = wf.readframes(wf.getnframes())
    if wf.getsampwidth() == 1:
        data = np.fromstring(f, dtype=np.int8)
    elif wf.getsampwidth() == 2:
        data = np.fromstring(f, dtype=np.int16)
    elif wf.getsampwidth() == 4:
        data = np.fromstring(f, dtype=np.int32)
    wdata = np.array(data,dtype='int32')
    return wdata,wf.getframerate()

def cov_var_mat(matrix):
    ones = np.ones((matrix.shape[0],matrix.shape[0]))
    dot = np.dot(ones,matrix)
    dev_scores = np.subtract(matrix,np.divide(dot,matrix.shape[0]))
    sum_squares = np.dot(dev_scores.T,dev_scores)
    covvar = np.divide(sum_squares,matrix.shape[0])
    return covvar

def get_mpc(path,filename,ncep):
    signal, rate = (get_audio_info(path))
    mfcc_feat,mpc_feat,spectrum  = mfcc(signal,fs=rate,nfft=512,nceps=ncep)
    feat = mpc_feat[:,:ncep]
    mean_feat = np.mean(feat, axis=0)
    covvar = cov_var_mat(feat)
    flat_covvar = []
    for i in range(0,covvar.shape[0]):
        flat_covvar.extend(list(covvar[i][:i+1]))
    feature_vector = []
    feature_vector.extend(list(mean_feat))
    feature_vector.extend(flat_covvar)
    feature_vector = np.array(feature_vector)
    data.append(feature_vector)
    labels.append(filename[0:-9])
    
def mpcfilter(path,ncep=592):
    explore(path,ncep)
    return data, labels

if __name__ == '__main__':
    mpcfilter('./test',33)
    




