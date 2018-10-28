#Import all the modules here
import pandas as pd

import processdata as procd
import importcsv as ic
import bayesian
import nn
import preprocessing
import svm
import arm as arm

#Insert main code here
if __name__ == '__main__':
    data = ic.separateImport()
    data = procd.fillData(data, fill_method="median")
    # in above function, fill_method has 'median', 'mode', and 'mean' options to fill data with the median, mode or mean

    testX, testY, trainX, trainY = procd.createTrainingSet(data)
    print(data)

    print("training set size: ", trainX.shape[0], " test set size: ", testX.shape[0] )

    #Sandbox your code here, before transfering it into your own python file

    min_sup = 0 #set it to be smth
    associateRuleMiningPredictions = arm.generate_rules(min_sup)
    print("Associate Rule Mining Predictions", associateRuleMiningPredictions)

    nnPredictions = nn.neuralNet(testX, testY, trainX, trainY, useTrainedModel = True)
    print("nnPredictions",nnPredictions)

    bayesPredictions = bayesian.naiveBayes(testX, testY, trainX, trainY)
    print("bayesPredictions",bayesPredictions)

    predictions, gs = svm.svmPredict(testX, testY, trainX, trainY, useTrainedModel = True)
    print("SVMpredictions", predictions)
    #gs is the grid search model that i use to find the best parameters for the svm.
    #It automatically uses k-fold cross validation to find the best parameters
    #we can call print(gs.best_params_) to determine what params were used for this model

    X_data, Y_data = preprocessing.createFullSet(data)
    optimal_n, X_reduced, pca, ss = preprocessing.manualSearchPCA(X_data)
    print('Best number of Components for pca:', optimal_n)
