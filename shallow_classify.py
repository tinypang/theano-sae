import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import RandomizedPCA
from mpc_filter import mpcfilter
import time
from format_dataset import relabel_data
from sklearn import svm
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from import_dataset import import_dataset
import re

def pltroc_curve(classifier,cv,data,labels,n):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(data[train], labels[train]).predict_proba(data[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(labels[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for {0} for {1} folds'.format(classifier,n))
    plt.legend(loc="lower right")
    plt.show()
    return

def test_classify(path,n,intype='mpc',SVC=False,linSVC=False,RNDforest=True,logR=False,indata=None,inlabels=None):
    log = open('results_shallow.txt','a+')
    log.write('dataset: {0}, input type: {1}, number of folds:{2}\n'.format(path, intype, n))
    if indata == None:
        ismir = re.compile('.*ismir.*') #identify dataset for label determination purposes
        gtzan = re.compile('.*GTZAN.*')
        if ismir.match(path) != None:
            dataset = 'ismir'
        elif gtzan.match(path) != None:
            dataset = 'gtzan'
        else:
            print 'unrecognised dataset type'
            sys.exit(
        data, labels = import_dataset(path,input_type=intype,nceps=33,pca=False,whiten=False,minmax=True,dataset=dataset)
        labels, labeldict = relabel_data(labels, 'classifier_label_mapping.txt')
        pairs = zip(label_dict.itervalues(), label_dict.iterkeys())
        pairs.sort()
        for i in pairs:
            log.write(str(i))
        log.write('\n')
        labels = np.array(labels)
    else:
        data = indata
        labels = inlabels
    skf = cross_validation.StratifiedKFold(labels, n_folds=n)
    svc = svm.SVC()
    linearsvc = svm.LinearSVC()
    rndforest = RandomForestClassifier()
    logregression = LogisticRegression(class_weight=auto)   #weight classes according to class frequency in dataset
    if RNDforest == True:
        print 'cross validating random forest'
        rndforest_scores =  cross_validation.cross_val_score(rndforest, data, labels, cv=skf)
        #pltroc_curve(rndforest,crossv,data,labels,n)
        print 'RndForest scores were {0} and average was {1}'.format(rndforest_scores, np.mean(rndforest_scores))
        log.write('RndForest scores were {0} and average was {1}\n'.format(rndforest_scores, np.mean(rndforest_scores)))
    if SVC == True:
        print 'cross validating svc'
        svc_scores =  cross_validation.cross_val_score(svc, data, labels, cv=skf)
        #pltroc_curve(svc,crossv,data,labels,n)
        print 'SVC scores were {0} and average was {1}'.format(svc_scores, np.mean(svc_scores))
        log.write('SVC scores were {0} and average was {1}\n'.format(svc_scores, np.mean(svc_scores)))
    if linSVC == True:
        print 'cross validating linear svc'
        linearsvc_scores =  cross_validation.cross_val_score(linearsvc, data, labels, cv=skf)
        #pltroc_curve(linearsvc,crossv,data,labels,n)
        print 'LinearSVC scores were {0} and average was {1}'.format(linearsvc_scores, np.mean(linearsvc_scores))
        log.write('LinearSVC scores were {0} and average was {1}\n'.format(linearsvc_scores, np.mean(linearsvc_scores)))
    if logR == True:
        print 'cross validating logistic regression'
        logR_scores =  cross_validation.cross_val_score(logregression, data, labels, cv=skf)
        #pltroc_curve(linearsvc,crossv,data,labels,n)
        print 'Logistic regression were {0} and average was {1}'.format(logR_scores, np.mean(logR_scores))
        log.write('Logistic regression scores were {0} and average was {1}\n'.format(logR_scores, np.mean(logR_scores)))
    log.close()
    return data,labels



if __name__ == '__main__':
    test_classify('./spectrogram/GTZAN_genre/3sec_50x20_gs',5,'spec',RNDforest=True)
