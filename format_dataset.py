from math import ceil

def split_dataset(data, labels,):
    print 'splitting data...'
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
                                
    mapping = open('label_mapping.txt','r+')    #record map[ings between integer label and string labels
    for i in label_dict.keys():
        mapping.write('{0}:{1}\n'.format(i,label_dict[i]))
    mapping.close()

    trainx,trainy,validx,validy,testx,testy = [],[],[],[],[],[]
    for i in range(0,len(data)):    #split data categories equally into 60% train, 10% valid, 30% test
        if i%10 in range(0,6,1):
            trainx.append(data[i])
            trainy.append(labels[i])
        elif i%10 == 6:
            validx.append(data[i])
            validy.append(labels[i])
        else:
            testx.append(data[i])
            testy.append(labels[i])
    print 'data has been split into train size {0}, validate size {1} and test size {2} sets'.format(len(trainx),len(validx),len(testx))
    return [(trainx,trainy), (validx,validy), (testx,testy), label_dict]    #return split data and label mapping
    
