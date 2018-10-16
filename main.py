#Import all the modules here
import pandas as pd

import bayesian
import processdata as procd
import importcsv as ic
import nn
#Insert main code here
if __name__ == '__main__':
    data = ic.basicImport()
    data = procd.cleanData(data)
    testX, testY, trainX, trainY = procd.createTrainingSet(data)
    print(data)

    print("training set size: ", trainX.shape[0], " test set size: ", testX.shape[0] )

    #Sandbox your code here, before transfering it into your own python file


    nnPredictions = nn.neuralNet(testX, testY, trainX, trainY, useTrainedModel = True)
    print("nnPredictions",nnPredictions)

    bayesPredictions = bayesian.naiveBayes(testX, testY, trainX, trainY)
    print("bayesPredictions",bayesPredictions)
