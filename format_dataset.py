from math import ceil
import numpy as np
import time
import random

def relabel_dataset(labels,data,label_dict=None):
    print 'relabelling data...'
    if label_dict==None:
        label_dict = {}
        n = 0
        for i in range(0,len(labels)):
            if labels[i] in label_dict:
                labels[i] = label_dict[labels[i]]
            else:
                print '{0} is a new label'.format(labels[i])    #relabel string labels as integer labels
                label_dict[labels[i]] = n
                labels[i] = n
                n +=1                                
    else:
        for i in range(0,len(labels)):
            labels[i] = label_dict[labels[i]]
    dataset = zip(data,labels)
    return dataset,label_dict

def split_dataset(dataset,n_classes,shuffle=False):
    count = {}
    for i in range(0,n_classes):     #initialise data split index count for each category
        count[i] = 0
    print 'splitting data...'
    pt1 = time.time()
    train = range(0,6,1)
    valid = [6]
    if shuffle == True:
        #random.shuffle(dataset)    #get random permutation of dataset
        modulos10 = range(0,10,1)
        train = random.sample(modulos10,6)
        for i in train:
            modulos10.remove(i)
        valid = random.sample(modulos10,1)    
    data, labels = zip(*dataset)
    #if testdata == None:
    trainx,trainy,validx,validy,testx,testy = [],[],[],[],[],[] #create empty arrays/lists for sets
    for j in range(0,len(data)):    #split data categories equally into 60% train, 10% valid, 30% test
        i = count[labels[j]]    #get split index for dataset based on its label
        if i%10 in train:    #assign 60% of data into train set
            trainx.append(data[j])
            trainy.append(labels[j])
        elif i%10 in valid:             #assign 10% of data into valid set
            validx.append(data[j])
            validy.append(labels[j])
        else:                       #assign 30% of data into test set
            testx.append(data[j])
            testy.append(labels[j])
        count[labels[j]] +=1    #update split index
    '''
    else:
        testx, testy = zip(*testdata)
        trainx,trainy,validx,validy = [],[],[],[]
        for j in range(0,len(data)):    #split data categories equally into 60% train, 10% valid, 30% test
            i = count[labels[j]]    #get split index for dataset based on its label
            if i%7 in range(0,5,1):    #assign 60% of data into train set
                trainx.append(data[j])
                trainy.append(labels[j])
            else:                       #assign 30% of data into test set
                testx.append(data[j])
                testy.append(labels[j])
            count[labels[j]] +=1    #update split index
    '''    
    trainx = np.array(trainx)
    validx = np.array(validx)
    testx = np.array(testx)
    pt2 = time.time()
    print 'data has been split into train size {0}, validate size {1} and test size {2} sets'.format(trainx.shape[0],validx.shape[0],testx.shape[0])
    print 'it took {0}sec'.format(pt2-pt1)
    return [(trainx,trainy), (validx,validy), (testx,testy)]

