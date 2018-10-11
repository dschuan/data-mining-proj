import numpy as np
import pandas as pd
import math

def cleanData(data):
    print('number of nans\n',data.isna().sum())

    data = data.drop(['ca','thal','slope'], axis=1)
    data = data.dropna()
    return data

def createTrainingSet(data):
    train_input = data.values

    X_data, Y_dataNum = train_input[:,:-1], train_input[:,-1]
    Y_dataNum = [isPositive(x) for x in Y_dataNum]


    idx = np.arange(X_data.shape[0])
    np.random.shuffle(idx)
    Y_data = np.array(Y_dataNum)
    X_data = X_data[idx]
    Y_data = Y_data[idx]

    m = 3* X_data.shape[0] // 10
    testX, testY =  X_data[:m], Y_data[:m]
    trainX, trainY = X_data[m:], Y_data[m:]

    return testX, testY, trainX, trainY

if __name__ == '__main__':
    print("Data Cleaning Output")
    print("training set size: ", trainX.shape[0], " test set size: ", testX.shape[0] )
