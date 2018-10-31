#Import all the modules here
import pandas as pd

import processdata as procd
import importcsv as ic
import bayesian
import nn
import preprocessing
import svm
from processResults import processResults
#Insert main code here

def reduceDimenTest():
	FILL_METHODS = ["mean","median","mode","none"]
	NUM_COMPONENTS = [4,8,12]
	for n_components in NUM_COMPONENTS:
		for filling in FILL_METHODS:
			print("**********************************now at ",filling, n_components)
			data = ic.separateImport()
			data = procd.fillData(data, fill_method=filling)
			# in above function, fill_method has 'median', 'mode', and 'mean' options to fill data with the median, mode or mean

			X_data, Y_data = preprocessing.createFullSet(data)
			X_data, pca, ss = preprocessing.performPCA(X_data, n_components)

			m = 3* X_data.shape[0] // 10
			testX, testY =  X_data[:m], Y_data[:m]
			trainX, trainY = X_data[m:], Y_data[m:]

			#print(data)

			print("training set size: ", trainX.shape[0], " test set size: ", testX.shape[0] )
			#Sandbox your code here, before transfering it into your own python file
			predictions = []
			methods = []
			#min_sup = 0 #set it to be smth
			#associateRuleMiningPredictions = arm.generate_rules(min_sup)
			#print("Associate Rule Mining Predictions", associateRuleMiningPredictions)

			nnPredictions = nn.neuralNet(testX, testY, trainX, trainY, useTrainedModel = True,modelName =  filling + str(n_components))
			print("nnPredictions",type(nnPredictions),nnPredictions)
			predictions.append(nnPredictions)
			methods.append("nnPredictions")

			bayesPredictions = bayesian.naiveBayes(testX, testY, trainX, trainY)
			print("bayesPredictions",type(bayesPredictions),bayesPredictions)
			predictions.append(bayesPredictions)
			methods.append("bayesPredictions")

			#gs is the grid search model that i use to find the best parameters for the svm.
			#It automatically uses k-fold cross validation to find the best parameters
			#we can call print(gs.best_params_) to determine what params were used for this model
			svmPredictions, clf = svm.svmPredict(testX, testY, trainX, trainY, filling+str(n_components), gridSearch=False)
			print("SVMpredictions", type(svmPredictions), svmPredictions)
			predictions.append(svmPredictions)
			methods.append("SVMpredictions")

			#best params precalculated to save time
			#res, best_params = dt.gridSearchWrapper(testX, testY, trainX, trainY)
			# best_params = {'n_estimators': 10, 'max_depth': 6, 'min_samples_split': 14}
			# randforestPred = dt.randomForestClassify(testX, testY, trainX, trainY, best_params)
			# print('Random forest',type(randforestPred), randforestPred)
			# predictions.append(randforestPred)
			# methods.append("Random forest")


			#ensemble method using a simple majority vote of all the classifiers.
			ensemblePred = []
			methods.append("Ensemble")
			for result in zip(*[item.tolist() for item in predictions]):
				ensemblePred.append(max(set(result), key=result.count))

			predictions.append(ensemblePred)
			print("ensemblePred", ensemblePred)


			for prediction, labels in zip(predictions,methods):
				print(labels,processResults(prediction,testY,filling,labels))


def fillMethodTest():
    data = ic.separateImport()
    data = procd.fillData(data, fill_method="median")
    # in above function, fill_method has 'median', 'mode', and 'mean' options to fill data with the median, mode or mean

    testX, testY, trainX, trainY = procd.createTrainingSet(data)
    print(data)

    print("training set size: ", trainX.shape[0], " test set size: ", testX.shape[0] )

    #Sandbox your code here, before transfering it into your own python file


    nnPredictions = nn.neuralNet(testX, testY, trainX, trainY, useTrainedModel = True)
    print("nnPredictions",nnPredictions)

    bayesPredictions = bayesian.naiveBayes(testX, testY, trainX, trainY)
    print("bayesPredictions",bayesPredictions)

    predictions, gs = svm.svmPredict(testX, testY, trainX, trainY, useTrainedModel = True)
    print("SVMpredictions", predictions)
    #gs is the grid search model that i use to find the best parameters for the svm. It automatically uses

    X_data, Y_data = preprocessing.createFullSet(data)
    optimal_n, X_reduced, pca, ss = preprocessing.manualSearchPCA(X_data)
    print('Best number of Components for pca:', optimal_n)

if __name__ == '__main__':
	reduceDimenTest()
