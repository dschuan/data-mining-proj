import processdata as procd
import importcsv as ic

import numpy as np

from pandas.tools.plotting import parallel_coordinates
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler




if __name__ ==  '__main__':

    data = ic.separateImport()
    data = procd.fillData(data, fill_method="median")
    data['prediction'] = data['prediction'].apply(lambda x: procd.isPositive(x))

    mms = MinMaxScaler()
    for header in list(data):
        data[header] = mms.fit_transform(data[header].values.reshape(-1, 1))

    parallel_coordinates(data, 'prediction', color=('red', 'green'))
    plt.show()
