import numpy as np
import pandas as pd
import math
import importcsv as ic
from sklearn.model_selection import train_test_split
def cleanData(data):
    #shows the number of missing data to determine if imputation will be helpful to improve prediction accuracy
    print('number of nans\n',data.isna().sum())
    data = data.drop(['ca','thal','slope'], axis=1)
    data = data.dropna()

    return data

def callMode(df):
    return df.mode()[0]

def callMean(df):
    return df.mean()

def callMedian(df):
    return df.median()

def callNone(df):
    return -1

def fillData(data_dict, fill_method = 'none', exclude_col = True):
    # Allows main code to choose from the four methods of cleaning data
    switcher = {
        'mode': callMode,
        'mean': callMean,
        'median': callMedian,
        'none': callNone
    }
    func = switcher.get(fill_method)
    output = pd.DataFrame()
    datas = list(data_dict.values())

    for df in datas:
        if exclude_col == True:
            df = df.drop(['ca','thal','slope'], axis=1)
        for column in df.iloc[:, :-1]:
            val = func(df[column])
            if fill_method != 'none':
                # imputes data with the fill method, if not 'none'
                df[column].fillna(val, inplace = True)
            else:
                # if fill method stays as 'none', just drop all NaN values
                df = df.dropna()
        #combines all the data together after imputation as a common dataset for analysis
        if (output.empty):
            output = df
        else:
            output = output.append(df, ignore_index=True)

    return output



# creates a data set
def createTrainingSet(data):
    train_input = data.values
    X_data, Y_dataNum = train_input[:,:-1], train_input[:,-1]
    Y_dataNum = [isPositive(x) for x in Y_dataNum]
    Y_data = np.array(Y_dataNum)
    m = 3* X_data.shape[0] // 10
    trainX, testX, trainY, testY = train_test_split(X_data, Y_data, test_size=m, random_state=42, stratify=Y_data)
    # train_input = data.values
    #
    # X_data, Y_dataNum = train_input[:,:-1], train_input[:,-1]
    # Y_dataNum = [isPositive(x) for x in Y_dataNum]
    # seed = 123
    # np.random.seed(seed)
    # idx = np.arange(X_data.shape[0])
    # np.random.shuffle(idx)
    # Y_data = np.array(Y_dataNum)
    # X_data = X_data[idx]
    # Y_data = Y_data[idx]
    # m = 3* X_data.shape[0] // 10
    # testX, testY =  X_data[:m], Y_data[:m]
    # trainX, trainY = X_data[m:], Y_data[m:]

    return testX, testY, trainX, trainY

def isPositive(x):
    if x > 0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    datadict = ic.separateImport()
    data = fillData(datadict, fill_method='median')
