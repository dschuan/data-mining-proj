import math
import tensorflow as tf
import numpy as np
import pylab as plt
import numpy
import pandas

sources = ["cleveland","long_beach","switzerland"]

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

# remove ca, thal, slope
REDUCED_LABELS = ['age',
'sex',
'cp',
'trestbps',
'chol',
'fbs',
'restecg',
'thalach',
'exang',
'oldpeak',
'prediction']

def separateImport():
    output = []
    for source in sources:
        temp = pandas.read_csv(source+".csv", delimiter="," , na_values = '?', names = LABELS )
        output.append(temp)
    # imports the data separately according to the file that it has been extracted from
    output = dict(zip(sources, output))
    return output

def basicImport():


    data = pandas.DataFrame()
    for source in sources:
        output = pandas.read_csv(source+".csv",delimiter=",",na_values  = '?',names =LABELS )

        if (data.empty):
            data = output
        else:
            data = data.append(output)

    return data



if __name__ == '__main__':
    data = separateImport()
