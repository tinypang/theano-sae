import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import RandomizedPCA
from mpc_filter import mpcfilter
import time

def import_mpc(path,nceps=33,scale=True,whiten=False,ncomps=100):
    pt0 = time.time()
    data, labels= mpcfilter(path,nceps)    #import audio files and extract mpc data and labels
    #output = open('gtzan_mpc_data.txt', 'w')
    #for i in data:
    #    output.write(str(i))
    #output.close()
    pt1 = time.time()
    print 'import and mpc coefficient extraction took time {0}'.format(pt1-pt0)
    data = np.array(data)   #convert python list of all img mpc data to a numpy array
    if scale == True:
        minmaxscaler = MinMaxScaler()   #create min max scale object to scale features between 0 and 1
        data = minmaxscaler.fit_transform(data) #apply min max scale to data
        pt2 = time.time()
        print 'scaling data to interval (0,1) took time {0}'.format(pt2-pt1)
    pt3 = time.time()
    if whiten == True:
       pca = RandomizedPCA(n_components=ncomps, whiten=True)  #create pca object to pca whiten features
       data = pca.fit_transform(data)   #apply pca whiten to data
       pt4 = time.time()
       print 'pca whiten took time {0}'.format(pt4-pt3)
    pt5 = time.time()
    print 'total time taken {0}'.format(pt5-pt0)
    return data, labels

if __name__ == '__main__':
    import_mpc('./test',33)

