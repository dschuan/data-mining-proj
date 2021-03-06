from sklearn.decomposition import PCA

import processdata as procd
import importcsv as ic

from processdata import isPositive

import numpy as np
from pylab import plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
from pathlib import Path
import preprocessing
from sklearn.model_selection import train_test_split

def gridSearchSVM(testX, testY, trainX, trainY):

    svc = SVC()
    param_grid = [
      {'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
      {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf', 'sigmoid']},
      {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.001, 0.0001], 'degree':[2], 'kernel': ['poly']},
      {'C': [1], 'gamma': [0.001], 'degree':[4], 'kernel': ['poly']}
    ]

    gs = GridSearchCV(svc, param_grid, verbose=2, n_jobs=4)
    gs.fit(trainX, trainY)
    score = gs.score(testX,testY)
    with open('svmModels/svm.pickle', 'wb') as fp:
        pickle.dump(gs, fp)
    return score

# def svmPredict(testX, testY, trainX, trainY, modelName, gridSearch = False):
#     if gridSearch:
#         gridSearchSVM(testX, testY, trainX, trainY)
#
#     savedModelPath = './svmModels/svm_' + modelName + '.pickle'
#     #look for the model
#     if Path(savedModelPath).is_file():
#         with open(savedModelPath, 'rb') as fp:
#             clf = pickle.load(fp)
#             predictions = clf.predict(testX)
#     else:
#         if not Path('svmModels/svm.pickle').is_file():
#             print('svm.pickle not found, run svmPredict with gridSearch = True')
#             raise FileNotFoundError
#         print(modelName, 'has not been trained before, loading svm.pickle(model with hyperparameters tuned) and training with trainX')
#         with open('svmModels/svm.pickle', 'rb') as fp:
#             gs = pickle.load(fp)
#         clf = SVC(**gs.best_params_)
#         # clf = SVC(C= 1, gamma= 0.001, kernel= 'rbf')
#         clf.fit(trainX, trainY)
#         predictions = clf.predict(testX)
#         #saving the trained model
#         with open(savedModelPath, 'wb') as fp:
#             pickle.dump(clf, fp)
#     return predictions, clf

def svmPredict(testX, testY, trainX, trainY, modelName, gridSearch = False):

    svc = SVC()
    param_grid = [
      {'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
      {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf', 'sigmoid']},
      {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.001, 0.0001], 'degree':[2,3], 'kernel': ['poly']}
    #   {'C': [1], 'gamma': [0.01], 'degree':[4], 'kernel': ['poly']}
    ]
    gs = GridSearchCV(svc, param_grid,verbose=4, n_jobs=4)
    gs.fit(trainX, trainY)
    predictions = gs.predict(testX)
    return predictions, gs


if __name__=='__main__':
    data = ic.separateImport()
    data = procd.fillData(data, fill_method="median")
    # testX, testY, trainX, trainY = procd.createTrainingSet(data)
    # X_data, Y_data = preprocessing.createFullSet(data)
    # X_data, pca, ss = preprocessing.performPCA(X_data, 10)
    # m = 3* X_data.shape[0] // 10
    # trainX, testX, trainY, testY = train_test_split(X_data, Y_data, test_size=m, random_state=42, stratify=Y_data)


    # gridSearchSVM(testX, testY, trainX, trainY)
    # testX, testY, trainX, trainY = procd.createTrainingSet(data)
    # score = gridSearchSVM(testX, testY, trainX, trainY )
    # print(score)
    # with open('svm.pickle', 'rb') as fp:
    #     gs = pickle.load(fp)
    #     print(gs.best_params_)
    # clf = SVC(**gs.best_params_)
    # clf.fit(trainX,trainY)
    # print(clf.score(testX, testY))
    # import pandas
    # results = pandas.DataFrame(gs.cv_results_)
    # results.to_csv('svmresults.csv')
    # print(results)

    X_data, Y_data = preprocessing.createFullSet(data)
    # X_data, pca, ss = preprocessing.performPCA(X_data, 8)
    m = 3* X_data.shape[0] // 10
    trainX, testX, trainY, testY = train_test_split(X_data, Y_data, test_size=m, random_state=42, stratify=Y_data)
    prediction, gs = svmPredict(testX, testY, trainX, trainY, '', gridSearch = False)
    sc = gs.score(testX, testY)
    print(sc, gs.best_params_)
