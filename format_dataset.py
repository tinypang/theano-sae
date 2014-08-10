from math import ceil

def split_dataset(data, labels,):
    print 'splitting data...'
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
        
    print 'data has been split into train, validate and test sets'
    return [(trainx,trainy), (validx,validy), (testx,testy)]
    
