#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from time import time
import logging
from operator import itemgetter
from sklearn.externals import joblib

from sklearn.preprocessing import OneHotEncoder,RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline ,Parallel
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import cross_val_score,train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,ExtraTreesClassifier,\
    AdaBoostClassifier,GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report,roc_auc_score,roc_curve
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV

"""

"""
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def get_category_index():
    path = "/home/lei/data/credit_prediction/features_type_copy.csv"
    indexlist = []
    for line in open(path, "r").readlines():
        indexlist.append(line)
    index = enumerate(indexlist)
    numidex = []
    cateindex = []
    for i in index:
        if i[1].strip() == 'numeric':
            numidex.append(i[0])
        else:
            cateindex.append(i[0])
    return numidex,cateindex

numIndex, cateIndex = get_category_index()
xPath = "train_x.csv"
xTestPath = "test_x.csv"
yPath = "train_y.csv"


def loadData(xPath,yPath=None):
    parentPath = "/home/lei/data/credit_prediction/"

    train_x = pd.read_csv(parentPath+xPath,dtype=np.float32)
    uid = train_x['uid'].values
    del train_x['uid']
    train_x.columns = np.array(range(1138))
    for idx in cateIndex:
        train_x[idx] = train_x[idx] + 1

    numFeatures = train_x[numIndex].values
    cateFeatures = train_x[cateIndex].values
    if yPath is not None:
        train_y = pd.read_csv(parentPath+yPath,dtype=np.float32)
        labels = train_y['y'].values
        return numFeatures, cateFeatures, labels
    elif yPath is None:
        return numFeatures,cateFeatures,uid

def featuresPreprocess(cateFeatures):
    """

    :param features:
    :return:
    """
    enc = OneHotEncoder(sparse=False)
    codedFeatures = enc.fit_transform(cateFeatures)
    return codedFeatures


def featuresSelect(features,labels=None):
    """

    :param features:
    :param labels:
    :return:
    """



def demensionReduction(numFeatures,cateFeatures):
    """

    :param numFeatures:
    :param labels:
    :return:
    """
    scaler = RobustScaler()
    scaledFeatures = scaler.fit_transform(numFeatures)
    pca = PCA(n_components=5)
    reducedFeatures = pca.fit_transform(scaledFeatures)
    allFeatures = np.concatenate((reducedFeatures,cateFeatures),axis=1)
    return allFeatures

def plotPCA(numfeatures,n_components=None):
    pca=PCA()
    varianceRatio = pca.explained_variance_ratio_
    accVar = varianceRatio
    for i,var in enumerate(accVar):
        if i==0:
            continue
        else:
            accVar[i]=accVar[i-1]+accVar[i]
    plt.clf()
    plt.xlabel('n_components')
    plt.ylabel('accumulation_variance')
    plt.plot(accVar)
    plt.show()

def paramsGridSearch(X,y,model=None):
    """

    :return:
    """
    #trainX,testX,trainY,testY = train_test_split(X,y,test_size=0.1,random_state=13)
    ada = AdaBoostClassifier(n_estimators=50)
    rf = RandomForestClassifier(n_estimators=100,oob_score=True,n_jobs=-1)
    gbt = GradientBoostingClassifier(n_estimators=50)

    start=time()
    print("starting grid search...")
    if model == "ada":
        paramsGrid = {}
        grid_search = GridSearchCV(ada,paramsGrid,scoring='roc_auc',cv=5,n_jobs=-1,verbose=1)
        grid_search.fit(X,y)
    elif model == "rf":
        paramsGrid = {"max_depth": [None],
                  "max_features": ['auto',None],
                  "min_samples_split": [1,3],
                  "min_samples_leaf": [1]}
        grid_search = GridSearchCV(rf,paramsGrid,scoring='roc_auc',cv=5,n_jobs=-1,verbose=1)
        grid_search.fit(X,y)
    elif model == "gbt":
        paramsGrid={}
        grid_search = GridSearchCV(gbt,paramsGrid,scoring='roc_auc',cv=5,n_jobs=-1,verbose=1)
        grid_search.fit(X,y)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))

    report(grid_search.grid_scores_)

    print "Score function of this estimator is-->", grid_search.scorer_
    print ""
    bestModel=grid_search.best_estimator_
    print("bestModel params is:{0}".format(bestModel.get_params()))
    #persist model
    #joblib.dump(bestModel,"/home/lei/data/credit_prediction/randomForest.pkl")

    return bestModel



def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def testClassification(model,testX,testY=None):

    print("This model params is:",model.get_params())

    scores = model.predict_proba(testX)
    return scores[:,1]


def balanceSamples(featureLabels):
    """

    :param numfeatures:
    :param catefeatures:
    :param labels:
    :return:
    """

    pos = featureLabels[featureLabels[:, -1]==1]
    neg = featureLabels[featureLabels[:, -1]==0]
    posDownSamples = pos[np.random.choice(pos.shape[0],5000,replace=False), :]
    negOverSamples = neg[np.random.choice(neg.shape[0],5000,replace=True), :]
    sampled = np.concatenate((posDownSamples,negOverSamples))
    return sampled

def featuresLabelUnion(numFeat,cateFeat,label=None):
    features = np.concatenate((numFeat, cateFeat),axis=1)
    if label is None:
        return features
    else:
        dlabels = np.array([label.tolist()]).T
        featureLabels = np.concatenate((features, dlabels),axis=1)
        return featureLabels



if __name__=="__main__":

    trainNumFeats,trainCateFeats,trainLabels = loadData(xPath,yPath)
    trainFeatLabels = featuresLabelUnion(trainNumFeats,trainCateFeats,trainLabels)

    print(trainFeatLabels.shape)

    sampled = balanceSamples(trainFeatLabels)
    print("Number of samples:%d; demension of features:%d" %
          (sampled.shape[0],sampled.shape[1]-1))
    samplednumFeat = sampled[:,range(1045)]
    sampledcateFeat = sampled[:,1045:-1]
    sampledLabels = sampled[:,-1]


    testNumFeats,testCateFeats,uid = loadData(xTestPath)


    pipeline = make_pipeline(RobustScaler(), PCA(n_components=5))
    pipeline.fit(trainNumFeats)
    reducedSampledNum = pipeline.transform(samplednumFeat)
    reducedTestNum = pipeline.transform(testNumFeats)
    reducedTrainNum = pipeline.transform(trainNumFeats)

    allTrainFeats = featuresLabelUnion(reducedTrainNum,trainCateFeats)
    allSampledFeats = featuresLabelUnion(reducedSampledNum,sampledcateFeat)
    allTestFeats = featuresLabelUnion(reducedTestNum,testCateFeats)

    print allSampledFeats.shape
    print allTestFeats.shape
    #paramsGridSearch(allTrainFeats, trainLabels,model='rf') #original unbalanced samples

    bestRF=paramsGridSearch(allSampledFeats, sampledLabels,'rf')#balanced sample
    score = testClassification(bestRF,allTestFeats)
    print(score)
    uid = [int(i) for i in uid.tolist()]
    print uid
    uidScore = zip(uid,score)
    df = pd.DataFrame(data=uidScore,columns=['uid','score'])
    df.to_csv("/home/lei/data/credit_prediction/uid_score.csv",index=False,header=True)
