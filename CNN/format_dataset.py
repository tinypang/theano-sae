from math import ceil
import numpy as np
import time

def relabel_data(labels,labelfile):
    print 'relabelling data...'
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
    mapping = open(labelfile,'r+')    #record map[ings between integer label and string labels
    for i in label_dict.keys():
        mapping.write('{0}:{1}\n'.format(i,label_dict[i]))
    mapping.close()
    return labels,label_dict

def split_dataset(data, labels,dimx,dimy):
    size = dimx*dimy
    labels,label_dict =  relabel_data(labels,'label_mapping.txt')
    print 'splitting data...'
    pt1 = time.time()    
    trainx,trainy,validx,validy,testx,testy = np.empty([0,size]),[],np.empty([0,size]),[],np.empty([0,size]),[]
    for i in range(0,len(data)):    #split data categories equally into 60% train, 10% valid, 30% test
        c = data[i]
        c.shape = (1,size)
        if i%10 in range(0,6,1):
            trainx = np.append(trainx,c,axis=0)
            trainy.append(labels[i])
        elif i%10 == 6:
            validx = np.append(validx,c,axis=0)
            validy.append(labels[i])
        else:
            testx = np.append(testx,c,axis=0)
            testy.append(labels[i])
    pt2 = time.time()
    print 'data has been split into train size {0}, validate size {1} and test size {2} sets'.format(trainx.shape[0],validx.shape[0],testx.shape[0])
    print 'it took {0}sec'.format(pt2-pt1)
    return [(trainx,trainy), (validx,validy), (testx,testy), label_dict]    #return split data and label mapping
    
