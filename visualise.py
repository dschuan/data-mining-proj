import processdata as procd
import importcsv as ic

import numpy as np

from pandas.tools.plotting import parallel_coordinates
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def parallelVisualise(data, measure, colors, measurename):
    mms = MinMaxScaler()
    for header in list(data):
        data[header] = mms.fit_transform(data[header].values.reshape(-1, 1))
    data[measurename] = measure
    print(data)
    parallel_coordinates(data, measurename, color =colors)
    plt.show()

if __name__ ==  '__main__':

    data = ic.separateImport()
    data = procd.fillData(data, fill_method="median")
    data['prediction'] = data['prediction'].apply(lambda x: procd.isPositive(x))

    parallelVisualise(data, measure=data['prediction'], colors=('red','green') , measurename='pred')
