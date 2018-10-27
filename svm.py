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
    #   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    #   {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf', 'sigmoid']},
    #   {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'degree':[2, 3, 4], 'kernel': ['poly']}
      {'C': [1, 10], 'kernel': ['linear']}
    ]
    gs = GridSearchCV(svc, param_grid, verbose=2, n_jobs=4)
    gs.fit(trainX, trainY)
    score = gs.score(testX,testY)
    with open('svm.pickle', 'wb') as fp:
        pickle.dump(gs, fp)
    return score

def svmPredict(testX, testY, trainX, trainY, useTrainedModel = True):
    if not useTrainedModel:
        gridSearchSVM(testX, testY, trainX, trainY)
    try:
        with open('svm.pickle', 'rb') as fp:
            gs = pickle.load(fp)
    except:
        print('\n*****************************svm.pickle not found, rerunning the gridsearch\n')
        gridSearchSVM(testX, testY, trainX, trainY)
    finally:
        with open('svm.pickle', 'rb') as fp:
            gs = pickle.load(fp)
    predictions = gs.predict(testX)
    return predictions, gs


if __name__=='__main__':
    data = ic.separateImport()
    data = procd.fillData(data, fill_method="median")
    testX, testY, trainX, trainY = procd.createTrainingSet(data)
    # score = gridSearchSVM(testX, testY, trainX, trainY )
    # with open('svm.pickle', 'rb') as fp:
    #     gs = pickle.load(fp)
    predictions, gs = svmPredict(testX, testY, trainX, trainY, False)
    print(predictions)
    print(testY)
    print(gs.best_params_)
# def predictSVM(testX, testY, trainX, trainY, useTrainedModel=False):
#     if useTrainedModel:
#         try:
#             gs=pickle.load('svm.pickle')
#             gs.predict(testX)
#         except:
#             continue
