from scipy import misc
import os
import numpy as np
import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
import time

standard_size = (380,240) #standard size for all spectrograms
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
def img_to_rgbmat(filename):    #convert image to array of rgba values of each pixel
    img = Image.open(filename)  #open image
    img = img.convert('L')    #convert image to greyscale
    img = img.resize(standard_size) #resize image to standard size
    img = list(img.getdata()) #get sequence object with pixel values for image and convert to a normal python list object
    img = map(list, img) #convert each sequence object row of pixels into a python list row
    img = np.array(img) #convert 2D list array into a numpy array
    return img

def flatten_img(img): #converts an (m,n) numpy array to an array of shape (1, m*n)
    s = img.shape[0] * img.shape[1] #calculate number of pixels in the img
    img_wide = img.reshape(1,s)     #reshape img to new size
    return img_wide[0]

def zero_mean(data,meandict):
    for i in data:
        for j in data[i]:
            data[i][j] = data[i][j]-meandict[j]    
    return data

def pca_whiten(path):
    data = []
    n = 0
    meandict = {}
    for i in 380:
        meandict[i] = 0
    pt0 = time.time()
    for filename in os.listdir(path):   #convert all images in a given path to rgb matrices, 
        imgfile = path + '/' + filename #flatten and append to list containing data for all images
        #pt1 = time.time()
        img = img_to_rgbmat(imgfile)
        #pt2 = time.time()
        img = flatten_img(img)
        #pt3 = time.time()
        for i in img:
            meandict[i] += img[i]/1000
        data.append(img)
        n+=1
        print n
    pt1 = time.time()
    print 'resize and import took time {0}'.format(pt1-pt0)
    data = zero_mean(data, meandict)
    pt2 = time.time()
    print 'zero mean took time {0}'.format(pt2-pt1)
    data = np.array(data)   #convert python list of all img data to a numpy array
    pt4 = time.time()
    print 'import and normalization took time {0}'.format(pt4 - pt0)
    pca = RandomizedPCA(n_components=10, whiten=True)
    X = pca.fit_transform(data)
    pt5 = time.time()
    print 'pca whitening took time {0}'.format(pt5 - pt4)
    print 'total time taken {0}'.format(pt5-pt0)
    return X
    #savefile = open('pca/pca_spectrograms_savefile.txt', 'w')
    #savefile.write(str(X))
    #savefile.close()

if __name__ == '__main__':
    #pca_whiten('./test')
    pca_whiten('../spectrogram/Spectrograms/Sample50')
