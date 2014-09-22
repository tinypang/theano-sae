from scipy import misc
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
import time
import math
from sklearn.preprocessing import MinMaxScaler

def flatten_img(filename): #opens and converts an (m,n) numpy array to an array of shape (1, m*n)
    img = Image.open(filename)  #open image
    img = list(img.getdata()) #get sequence object with pixel values for image and convert to a normal python list object
    #   img = map(list, img) #convert each sequence object row of pixels into a python list row
    #img = np.array(img) #convert 2D list array into a numpy array
    #   s = img.shape[0] * img.shape[1] #calculate number of pixels in the 
    #   img_wide = img.reshape(1,s)     #reshape img to new size
    return img  #return image as an 1xn araay

def minmaxscale(data):
    pt1 = time.time()
    minmaxscaler = MinMaxScaler()   #create min max scale object to scale features between 0 and 1
    data = minmaxscaler.fit_transform(data) #apply min max scale to data
    pt2 = time.time()
    print 'scaling data to interval (0,1) took time {0}'.format(pt2-pt1)
    return data 

def zero_mean(data,meandict):   #normalizes each feature to a zero mean
    for i in range(0,len(data)):      #for each feature vector
        for j in range(0,len(data[i])):   #subtract the mean of the feature from the feature value
            data[i][j] = data[i][j]-meandict[j]    
    return data #return mean normalized data

def import_data(path, dimx, dimy): 
    data,labels = [],[] #intiate arrays to store data and labels
    n = 0   #initiate integer variable to track data import
    pt0 = time.time()
    for filename in os.listdir(path):   #for all images in dataset flatten and append to list containing data for all images 
        imgfile = path + '/' + filename #create file address from file path and file name
        img = flatten_img(imgfile)  #flatten image
        data.append(img)
        labels.append(filename[0:-28])
        n+=1
        print n
    pt1 = time.time()
    print 'resize and import took time {0}'.format(pt1-pt0)
    return data, labels
    
def pca(data,ncomp=100,whiten=False):
    pt4 = time.time()
    print 'import and normalization took time {0}'.format(pt4 - pt0)
    if whiten == True:   #if data needs to be pca whitened, whiten data
       pca = RandomizedPCA(n_components=ncomp, whiten=True)  #create pca object to pca whiten features
       X = pca.fit_transform(data)
    else:
       X = data         #else return data as is
    pt5 = time.time()
    print 'array cast and pca whitening took time {0}'.format(pt5 - pt2)
    print 'total time taken {0}'.format(pt5-pt0)
    return X

def import_preprocess_data(path, dimx, dimy, ncomp=100, pca = False, whiten=False):
    data, labels = import_data(path, dimx, dimy)
    data = np.array(data)   #convert python list of all img data to a numpy array
    data = minmaxscale(data)
    if pca ==True:
        data = pca(data,ncomp,whiten)
    return data, labels


if __name__ == '__main__':
    pca('./spectrogram/test',38,24)
    #pca('./spectrogram/preprocessed')

