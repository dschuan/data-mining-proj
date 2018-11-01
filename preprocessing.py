from sklearn.decomposition import PCA

import processdata as procd
import importcsv as ic

from processdata import isPositive
from sklearn.preprocessing import StandardScaler
import numpy as np
from pylab import plt
from sklearn.model_selection import GridSearchCV


def createFullSet(data):
    #returns data set without train test split. IE only
    train_input = data.values
    X_data, Y_dataNum = train_input[:,:-1], train_input[:,-1]
    Y_dataNum = [isPositive(x) for x in Y_dataNum]
    Y_data = np.array(Y_dataNum)
    # idx = np.arange(X_data.shape[0])
    # seed = 123
    # np.random.seed(seed)
    # np.random.shuffle(idx)
    # X_data = X_data[idx]
    # Y_data = Y_data[idx]

    return X_data, Y_data

def gridSearchPCA(X_data):
    #returns the best parameter that grid search finds on PCA
    #Does cross validation with random train splits making this comparison unreliable
    ss = StandardScaler()
    X_data = ss.fit_transform(X_data)
    pca = PCA()

    gs = GridSearchCV(pca, {'n_components':range(1, 14)}, cv=0)
    gs.fit (X_data)
    X_reduced = gs.transform (X_data)
    return gs.best_params_

def manualSearchPCA(X_data):
    ss = StandardScaler()
    X_data = ss.fit_transform(X_data)
    pca = PCA()
    scores = []
    dimensions = X_data.shape[1]
    for i in range(1,dimensions):
        pca.set_params(n_components=i)
        pca.fit(X_data)
        scores.append(pca.score(X_data))
    optimal_n = scores.index(max(scores)) + 1
    pca.set_params(n_components=optimal_n)
    X_reduced = pca.fit_transform(X_data)
    return optimal_n, X_reduced, pca, ss

def performPCA(X_data, n_components):
    ss = StandardScaler()
    X_data = ss.fit_transform(X_data)
    pca=PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_data)
    return X_reduced, pca, ss


if __name__ == '__main__':

    data = ic.separateImport()
    data = procd.fillData(data, fill_method="median")

    X_data, Y_data = createFullSet(data)
    #we scale the data to perform a more accurate pca decomposition
    #can 'unscale' using ss.inverse_transform(X_data)

    X_reduced, pca, ss = performPCA(X_data, 2)
    print(pca.explained_variance_ratio_)

    data_reduced = np.concatenate((X_reduced, Y_data.reshape(626,1)), axis=1)
    print(data_reduced)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [0, 1]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = data_reduced[:,-1] == target
        ax.scatter(data_reduced[indicesToKeep, 0]
                   , data_reduced[indicesToKeep, 1]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()
