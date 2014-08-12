from math import ceil

def split_dataset(data, labels,):
    print 'splitting data...'
    label_dict = {}
    n = 0
    for i in range(0,len(labels)):
        if labels[i] in label_dict:
            labels[i] = label_dict[labels[i]]
        else:
            label_dict[labels[i]] = n
            labels[i] = n
            n +=1
                        
    mapping = open('label_mapping.txt','r+')
    for i in label_dict.keys():
        mapping.write('{0},{1}'.format(i,label_dict[i]))
    mapping.close()

    trainx,trainy,validx,validy,testx,testy = [],[],[],[],[],[]
    for i in range(0,len(data)):
        if i%4 == 0 or i%4 == 1:
            trainx.append(data[i])
            trainy.append(labels[i])
        elif i%4 == 2:
            validx.append(data[i])
            validy.append(labels[i])
        else:
            testx.append(data[i])
            testy.append(labels[i])
    #print len(trainx),len(trainy)
    #print len(validx),len(validy)
    #print len(testx),len(testy)
    print 'data has been split into train, validate and test sets'
    return [(trainx,trainy), (validx,validy), (testx,testy)]
    
