from sklearn.decomposition import PCA

import processdata as procd
import importcsv as ic

from processdata import isPositive

import numpy as np
from pylab import plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

data = ic.separateImport()
data = procd.fillData(data, fill_method="median")

testX, testY, trainX, trainY = procd.createTrainingSet(data)

svc = SVC()
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf', 'sigmoid']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'degree':[2, 3, 4], 'kernel': ['poly']}
]
gs = GridSearchCV(svc, param_grid, verbose=2)
gs.fit(trainX, trainY)
score = gs.score(testX,testY)
print(score)
