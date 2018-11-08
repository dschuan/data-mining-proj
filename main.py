#Import all the modules here
import pandas as pd
import decisiontree as dt
import processdata as procd
import importcsv as ic
import bayesian
import nn
import preprocessing
import svm
import pickle
import DBSCAN
from processResults import processResults, generateGraphs,generateGraphsSingle
from collections import defaultdict
from sklearn.model_selection import train_test_split
#Insert main code here

def reduceDimenTest(extramodelNames = ''):
	FILL_METHODS = ["mean","median","mode","none"]

	#space to search in where num_components refer to the dimensionality of the processed dataset
	NUM_COMPONENTS = [4,8,10]
	processedResults = defaultdict(lambda: [])



	for n_components in NUM_COMPONENTS:
		for filling in FILL_METHODS:
			print("**********************************now at ",filling, n_components)

			# in above function, fill_method has 'median', 'mode', and 'mean' options to fill data with the median, mode or mean
			data = ic.separateImport()
			data = procd.fillData(data, fill_method=filling)
			X_data, Y_data = preprocessing.createFullSet(data)
			X_data, pca, ss = preprocessing.performPCA(X_data, n_components)
			m = 3* X_data.shape[0] // 10
			trainX, testX, trainY, testY = train_test_split(X_data, Y_data, test_size=m, random_state=42, stratify=Y_data)

			#print(data)

			print("training set size: ", trainX.shape[0], " test set size: ", testX.shape[0] )
			#Sandbox your code here, before transfering it into your own python file
			predictions = []
			methods = []
			#min_sup = 0 #set it to be smth
			#associateRuleMiningPredictions = arm.generate_rules(min_sup)
			#print("Associate Rule Mining Predictions", associateRuleMiningPredictions)



			nnPredictions = nn.neuralNet(testX, testY, trainX, trainY, useTrainedModel = True,modelName =  filling + str(n_components) + extramodelNames)
			#print("nnPredictions",type(nnPredictions),nnPredictions)
			predictions.append(nnPredictions)
			methods.append("nnPredictions")



			bayesPredictions = bayesian.naiveBayes(testX, testY, trainX, trainY)
			#print("bayesPredictions",type(bayesPredictions),bayesPredictions)
			predictions.append(bayesPredictions)
			methods.append("bayesPredictions")



			#gs is the grid search model that i use to find the best parameters for the svm.
			#It automatically uses k-fold cross validation to find the best parameters
			#we can call print(gs.best_params_) to determine what params were used for this model
			svmPredictions, clf = svm.svmPredict(testX, testY, trainX, trainY, filling+str(n_components)+extramodelNames, gridSearch=False)
			#print("SVMpredictions", type(svmPredictions), svmPredictions)
			predictions.append(svmPredictions)
			methods.append("SVMpredictions")



			#best hyperparams precalculated using grid search model to save time
			#res, best_params = dt.gridSearchWrapper(testX, testY, trainX, trainY)
			best_params = {'n_estimators': 10, 'max_depth': 6, 'min_samples_split': 14}
			randforestPred = dt.randomForestClassify(testX, testY, trainX, trainY, best_params)
			predictions.append(randforestPred)
			methods.append("randforest")
			# methods.append("Random forest")

			#ensemble method using a simple majority vote of all the classifiers.
			ensemblePred = []

			for result in zip(*[item.tolist() for item in predictions]):
				ensemblePred.append(max(set(result), key=result.count))

			predictions.append(ensemblePred)
			methods.append("Ensemble")
			#print("ensemblePred", ensemblePred)


			for prediction, labels in zip(predictions,methods):
				result = processResults(prediction,testY,filling,labels)
				result["n_components"] = n_components
				result["filling"] = filling
				processedResults[labels].append(result)
	generateGraphsSingle(processedResults,FILL_METHODS)
	generateGraphs(processedResults,FILL_METHODS)



if __name__ == '__main__':
	# reduceDimenTest('tenDim')
	DBSCAN.perform_dbscan()
