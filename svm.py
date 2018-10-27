from sklearn.decomposition import PCA

import processdata as procd
import importcsv as ic

from processdata import isPositive

import numpy as np
from pylab import plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle

def gridSearchSVM(testX, testY, trainX, trainY):

    svc = SVC()
    param_grid = [
      {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf', 'sigmoid']},
      {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'degree':[2, 3, 4], 'kernel': ['poly']}
    ]
    gs = GridSearchCV(svc, param_grid, verbose=2)
    gs.fit(trainX, trainY)
    score = gs.score(testX,testY)
    pickle.dump(gs, 'svm.pickle')
    return score
if __name__=='__main__':
    data = ic.separateImport()
    data = procd.fillData(data, fill_method="median")
    testX, testY, trainX, trainY = procd.createTrainingSet(data)
    score = gridSearchSVM(testX, testY, trainX, trainY )
    print(score)
# def predictSVM(testX, testY, trainX, trainY, useTrainedModel=False):
#     if useTrainedModel:
#         try:
#             gs=pickle.load('svm.pickle')
#             gs.predict(testX)
#         except:
#             continue
