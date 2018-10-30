from sklearn import metrics
def processResults(prediction,testY,filling,method):


    #saveGraphs(prediction,testY,filling,method)

    return analyzeResults(prediction,testY,filling,method)





#def saveGraphs(prediction,testY,filling,method):


def analyzeResults(prediction,testY,filling,method):

    return metrics.classification_report(testY, prediction)
