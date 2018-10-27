#Import all the modules here
import pandas as pd

import processdata as procd
import importcsv as ic
import bayesian
import nn
import preprocessing
import svm

#Insert main code here
if __name__ == '__main__':
    data = ic.separateImport()
    data = procd.fillData(data, fill_method="median")
    # in above function, fill_method has 'median', 'mode', and 'mean' options to fill data with the median, mode or mean

    testX, testY, trainX, trainY = procd.createTrainingSet(data)
    print(data)

    print("training set size: ", trainX.shape[0], " test set size: ", testX.shape[0] )

    #Sandbox your code here, before transfering it into your own python file


    nnPredictions = nn.neuralNet(testX, testY, trainX, trainY, useTrainedModel = True)
    print("nnPredictions",nnPredictions)

    bayesPredictions = bayesian.naiveBayes(testX, testY, trainX, trainY)
    print("bayesPredictions",bayesPredictions)

    predictions, gs = svm.svmPredict(testX, testY, trainX, trainY, useTrainedModel = True)
    print("SVMpredictions", predictions)
    
    X_data, Y_data = preprocessing.createFullSet(data)
    optimal_n, X_reduced, pca, ss = preprocessing.manualSearchPCA(X_data)
    print('Best number of Components for pca:', optimal_n)
