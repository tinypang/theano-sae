from math import ceil
import numpy as np
import time

def relabel_data(labels,labelfile):
    print 'relabelling data...'
    label_dict = {}
    count = {}
    n = 0
    for i in range(0,len(labels)):
        if labels[i] in label_dict:
            labels[i] = label_dict[labels[i]]
        else:
            print '{0} is a new label'.format(labels[i])    #relabel string labels as integer labels
            label_dict[labels[i]] = n
            labels[i] = n
            n +=1
    mapping = open(labelfile,'r+')    #record mappings between integer label and string labels
    for i in label_dict.keys():
        mapping.write('{0}:{1}\n'.format(i,label_dict[i]))
    mapping.close()
    return labels,label_dict

def split_dataset(data, labels,size):
    labels,label_dict =  relabel_data(labels,'label_mapping.txt')
    print 'splitting data...'
    pt1 = time.time()    
    trainx,trainy,validx,validy,testx,testy = np.empty([0,size]),[],np.empty([0,size]),[],np.empty([0,size]),[]
    for i in range(0,len(data)):    #split data categories equally into 60% train, 10% valid, 30% test
        if i%10 in range(0,6,1):
            trainx.append(data[i])
            trainy.append(labels[i])
        elif i%10 == 6:             #assign 10% of data into valid set
            validx.append(data[i])
            validy.append(labels[i])
        else:                       #assign 30% of data into test set
            testx.append(data[i])
            testy.append(labels[i])
        data[i] = 0
    pt2 = time.time()
    print 'data has been split into train size {0}, validate size {1} and test size {2} sets'.format(trainx.shape[0],validx.shape[0],testx.shape[0])
    print 'it took {0}sec'.format(pt2-pt1)
    return [(trainx,trainy), (validx,validy), (testx,testy), label_dict]    #return split data and label mapping
 
def split_dataset_ismirg(data, labels,size):
    labels,label_dict =  relabel_data(labels,'label_mapping.txt')   #relabel data
    count = {}
    for i in label_dict.values():     #initialise data split index count for each category
        count[i] = 0
    print 'splitting data...'
    pt1 = time.time()
    #trainx,trainy,validx,validy,testx,testy = np.empty([0,size]),[],np.empty([0,size]),[],np.empty([0,size]),[] #create empty arrays/lists for sets
    trainx,trainy,validx,validy,testx,testy = [],[],[],[],[],[] #create empty arrays/lists for sets
    for j in range(0,len(data)):    #split data categories equally into 60% train, 10% valid, 30% test
        i = count[labels[j]]    #get split index for dataset based on its label
        if i%10 in range(0,6,1):    #assign 60% of data into train set
            trainx.append(data[j])
            trainy.append(labels[j])
        elif i%10 == 6:             #assign 10% of data into valid set
            validx.append(data[j])
            validy.append(labels[j])
        else:                       #assign 30% of data into test set
            testx.append(data[j])
            testy.append(labels[j])
        count[labels[j]] +=1    #update split index
        data[j] = 0             #clear categorised data 
    trainx = np.array(trainx)
    validx = np.array(validx)
    testx = np.array(testx)
    pt2 = time.time()
    print 'data has been split into train size {0}, validate size {1} and test size {2} sets'.format(trainx.shape[0],validx.shape[0],testx.shape[0])
    print 'it took {0}sec'.format(pt2-pt1)
    return [(trainx,trainy), (validx,validy), (testx,testy), label_dict]    #return split data and label mapping
 
