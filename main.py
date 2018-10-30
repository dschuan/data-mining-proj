#Import all the modules here
import pandas as pd
from processResults import processResults
from sklearn.ensemble import RandomForestClassifier
import processdata as procd
import importcsv as ic
import bayesian
import nn
import preprocessing
import svm
import arm as arm
import decisiontree as dt

#Insert main code here
if __name__ == '__main__':


	FILL_METHODS = ["mean","median","mode","none"]

	for filling in FILL_METHODS:
		print("**********************************now at ",filling)
		data = ic.separateImport()
		data = procd.fillData(data, fill_method=filling)
		# in above function, fill_method has 'median', 'mode', and 'mean' options to fill data with the median, mode or mean

		testX, testY, trainX, trainY = procd.createTrainingSet(data)
		#print(data)

		print("training set size: ", trainX.shape[0], " test set size: ", testX.shape[0] )

		X_data, Y_data = preprocessing.createFullSet(data)
		optimal_n, X_reduced, pca, ss = preprocessing.manualSearchPCA(X_data)
		print('Best number of Components for pca:', optimal_n)

		#Sandbox your code here, before transfering it into your own python file
		predictions = []
		methods = []
		#min_sup = 0 #set it to be smth
		#associateRuleMiningPredictions = arm.generate_rules(min_sup)
		#print("Associate Rule Mining Predictions", associateRuleMiningPredictions)

		nnPredictions = nn.neuralNet(testX, testY, trainX, trainY, useTrainedModel = True,modelName = filling)
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
		svmPredictions, clf = svm.svmPredict(testX, testY, trainX, trainY, filling, gridSearch=False)
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
