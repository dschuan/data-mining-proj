#Import all the modules here
import pandas as pd


import processdata as procd
import importcsv as ic
#Insert main code here
if __name__ == '__main__':
    data = ic.basicImport()
    data = procd.cleanData(data)
    testX, testY, trainX, trainY = procd.createTrainingSet(data)
    #Sandbox your code here, before transfering it into your own python file








    print(data)
    print("training set size: ", trainX.shape[0], " test set size: ", testX.shape[0] )
