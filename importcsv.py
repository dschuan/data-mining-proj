import math
import tensorflow as tf
import numpy as np
import pylab as plt
import numpy
import pandas

# geographical locations/ file name source of our data, in csv format
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

# removed ca, thal, slope from the original 14 attributes

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

# keeps the data separated according to geographical location, so that imputation can be done with lower bias

def separateImport():
    output = []
    for source in sources:
        temp = pandas.read_csv(source+".csv", delimiter="," , na_values = '?', names = LABELS )
        output.append(temp)
    # imports the data separately according to the file that it has been extracted from
    output = dict(zip(sources, output))
    return output


#imports all the data together
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
