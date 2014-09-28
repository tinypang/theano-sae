import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import RandomizedPCA
from mpc_filter import mpcfilter
import time
from format_dataset import relabel_data
from sklearn import svm
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

def test_classify(path,n,SVC=False,linSVC=False,RNDforest=True,indata=None,inlabels=None):
    if indata == None:
        data, labels = import_mpc(path,33,scale=True,whiten=False)
        labels = np.array(relabel_data(labels, 'classifier_label_mapping.txt'))
    else:
        data = indata
        labels = inlabels
    crossv = cross_validation.StratifiedKFold(labels, n_folds=n)
    svc = svm.SVC()
    linearsvc = svm.LinearSVC()
    rndforest = RandomForestClassifier()
    if RNDforest == True:
        print 'cross validating random forest'
        rndforest_scores =  cross_validation.cross_val_score(rndforest, data, labels, cv=n)
        #pltroc_curve(rndforest,crossv,data,labels,n)
        print 'RndForest scores were {0} and average was {1}'.format(rndforest_scores, np.mean(rndforest_scores))
    if SVC == True:
        print 'cross validating svc'
        svc_scores =  cross_validation.cross_val_score(svc, data, labels, cv=n)
        #pltroc_curve(svc,crossv,data,labels,n)
        print 'SVC scores were {0} and average was {1}'.format(svc_scores, np.mean(svc_scores))
    if linSVC == True:
        print 'cross validating linear svc'
        linearsvc_scores =  cross_validation.cross_val_score(linearsvc, data, labels, cv=n)
        #pltroc_curve(linearsvc,crossv,data,labels,n)
        print 'LinearSVC scores were {0} and average was {1}'.format(linearsvc_scores, np.mean(linearsvc_scores))
    '''

    '''
    return data,labels
    
        

if __name__ == '__main__':
    test_classify('./audio_snippets/GTZAN_3sec',2)

