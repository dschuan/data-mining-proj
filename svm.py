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



def gridSearchSVM(testX, testY, trainX, trainY):

    svc = SVC()
    param_grid = [
      {'C': [1, 10, 100], 'kernel': ['linear']},
      {'C': [1, 10, 100], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf', 'sigmoid']},
      {'C': [1, 10, 100], 'gamma': [0.01, 0.001, 0.0001], 'degree':[2, 3], 'kernel': ['poly']},
      {'C': [1], 'gamma': [0.01], 'degree':[4], 'kernel': ['poly']}
    ]
    gs = GridSearchCV(svc, param_grid, verbose=2, n_jobs=4)
    gs.fit(trainX, trainY)
    score = gs.score(testX,testY)
    with open('svmModels/svm.pickle', 'wb') as fp:
        pickle.dump(gs, fp)
    return score

def svmPredict(testX, testY, trainX, trainY, modelName, gridSearch = False):
    if gridSearch:
        gridSearchSVM(testX, testY, trainX, trainY)

    savedModelPath = './svmModels/svm_' + modelName + '.pickle'
    #look for the model
    if Path(savedModelPath).is_file():
        with open(savedModelPath, 'rb') as fp:
            clf = pickle.load(fp)
            predictions = clf.predict(testX)
    else:
        if not Path('svmModels/svm.pickle').is_file():
            print('svm.pickle not found, run svmPredict with gridSearch = True')
            raise FileNotFoundError
        print(modelName, 'has not been trained before, loading svm.pickle(model with hyperparameters tuned) and training with trainX')
        with open('svmModels/svm.pickle', 'rb') as fp:
            gs = pickle.load(fp)
        # clf = SVC(**gs.best_params_)
        clf = SVC(C= 1, gamma= 0.001, kernel= 'rbf')
        clf.fit(trainX, trainY)
        predictions = clf.predict(testX)
        #saving the trained model
        with open(savedModelPath, 'wb') as fp:
            pickle.dump(clf, fp)
    return predictions, clf


if __name__=='__main__':
    # data = ic.separateImport()
    # data = procd.fillData(data, fill_method="median")
    # testX, testY, trainX, trainY = procd.createTrainingSet(data)
    # # score = gridSearchSVM(testX, testY, trainX, trainY )
    # # with open('svmModels/svm.pickle', 'rb') as fp:
    # #     gs = pickle.load(fp)
    # from pandas.tools.plotting import parallel_coordinates
    # from matplotlib import pyplot as plt
    #
    # print(data)
    # parallel_coordinates(data, 'prediction')
    # plt.show()
    with open('svmModels/svm.pickle', 'rb') as fp:
        gs = pickle.load(fp)
    print(gs.best_params_)
