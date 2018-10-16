from sklearn import datasets
from sklearn import metrics

from sklearn.naive_bayes import GaussianNB

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import numpy
import pandas


sources = ["cleveland","long_beach","switzerland"]
#sources = ["cleveland"]



LABELS = ['age',
'sex',
'cp',
'trestbps',
'chol',
'fbs',
'restecg',
'thalach',
'exang',
'oldpeak',
'slope',
'ca',
'thal',
'prediction']

NUM_CLASSES = 2
seed = 123
np.random.seed(seed)

def naiveBayes(testX, testY, trainX, trainY, showMetrics = False):

    print("training set size: ", trainX.shape[0], " test set size: ", testX.shape[0] )

    model = GaussianNB()

    model.fit(trainX, trainY)


    testYPrediction = model.predict(testX)

    if(showMetrics):
        print(metrics.classification_report(testY, testYPrediction))

        print(metrics.confusion_matrix(testY, testYPrediction))

    return testYPrediction
