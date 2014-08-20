from scipy import misc
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
import time
import math

'''
def normalize(arr): #linearly nromalize each pixel in an image
    # Do not touch the alpha channel
    for i in range(3):  #for rgb values
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            np.subtract(arr[...,i],minval)
            np.multiply(arr[...,i],(255.0/(maxval-minval)))
    return arr
'''

def flatten_img(filename): #opens and converts an (m,n) numpy array to an array of shape (1, m*n)
    img = Image.open(filename)  #open image
    img = list(img.getdata()) #get sequence object with pixel values for image and convert to a normal python list object
    #   img = map(list, img) #convert each sequence object row of pixels into a python list row
    #img = np.array(img) #convert 2D list array into a numpy array
    #   s = img.shape[0] * img.shape[1] #calculate number of pixels in the 
    #   img_wide = img.reshape(1,s)     #reshape img to new size
    return img  #return image as an 1xn araay

def zero_mean(data,meandict):   #normalizes each feature to a zero mean
    for i in range(0,len(data)):      #for each feature vector
        for j in range(0,len(data[i])):   #subtract the mean of the feature from the feature value
            data[i][j] = data[i][j]-meandict[j]    
    return data #return mean normalized data

def pca(path,dimx,dimy,ncomp=100,whiten=False):
    data,labels = [],[] #intiate arrays to store data and labels
    n = 0   #initiate integer variable to track data import
    meandict = {}   #initiate dictionary to store the mean of each feature i
    for i in range(0,dimx*dimy):
        meandict[i] = 0
    pt0 = time.time()
    for filename in os.listdir(path):   #for all images in dataset flatten and append to list containing data for all images 
        imgfile = path + '/' + filename #create file address from file path and file name
        img = flatten_img(imgfile)  #flatten image
        for i in range(0,len(img)):   #add each feature value to mean dictionary
            meandict[i] += img[i]/1000
        data.append(img)
        labels.append(filename[0:-27])
        n+=1
        print n
    pt1 = time.time()
    print 'resize and import took time {0}'.format(pt1-pt0)
    #data = zero_mean(data, meandict)    #normalize each feature to a zero mean
    meandict = []
    pt2 = time.time()
    print 'zero mean took time {0}'.format(pt2-pt1)
    data = np.array(data)   #convert python list of all img data to a numpy array
    #data = pca_whiten(data) #perform pca whitening on datased
    #print data.shape
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
    return X, labels

'''
def pca_whiten(x):
    print 'performing pca'
    t0 = time.time()
    print 'finding sigma...'
    sigma = np.divide(np.dot(x.transpose(),x),x.shape[1])    #find covariance matrix for input data
    print 'converting sigma from array to matrix...'
    sigma = np.matrix(sigma)    #turn covariance matrix into matrix data type
    print 'performing svd on sigma'
    U, s, V = np.linalg.svd(sigma)  #find svd decomposition of covariance matrix
    V = []      #deallocate V memory as it is U transpose and not used
    sigma = []  #deallocate sigma memory
    print 'finding k...'
    k = -1  #initialise k, the number of dimensions to reduce to
    m = data.shape[0]   #set m, the max valeu of k
    total_singular = 0  #calculate the cumulative sum of all singular values
    for i in s:
        total_singular += i
    while var <=0.99 or k<=m:   #while variance is less than 99% or k less than m
        k += 1                  #calculate the lowest value of k that when the first k singular
        total_k = 0             #values are summed, the variance is greater than 99%
        for i in range(0,k+1):
            total_k += s[i]
        var = total_k / total_singular
    if k > m:                   #if k is found to be greater than m, exit with an error msg
        print 'error! k >m'
        exit
    print 'k was found to be {0}'.format(k)
    print 'performing pca dimensionality reduction from {0} to {1} dimensions...'.format(m,k) 
    z = np.dot(U[:,0:k+1].T,x)  #perform pca by multiplying first k columns of U with input data
    epsilon  = 0.1  #set pca whitening value, 0 means normal pca
    print 'epsilon set to {0}, performing pca whitening...'.format(epsilon)
    pca_white = np.dot(np.diag((1./math.sqrt(np.add(s,epsilon))),z)) #pca whiten data by dividing by singular vlaues
    print 'pca whitening completed, casting to array and returning'
    pca_white = np.array(pca_white) #cast pca whitened data back to array
    t1 = time.time()
    print 'total time taken was {0}sec'.format(t1 - t0)
    return pca_white               
'''


if __name__ == '__main__':
    pca('./spectrogram/test',38,24)
    #pca('./spectrogram/preprocessed')

