from sklearn.ensemble import RandomForestClassifier
import importcsv as ic
import processdata as procd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

import pandas as pd
"""
	Creates the possible iterations for the relevant hyperparameters.
	Only number of trees, maximum tree depth, and 
"""
sample_split = range(2, 16, 2)
n_estimator = range(10, 310, 100)
max_depth = range(3, 24, 3)

param_grid = {
	'min_samples_split': sample_split,
	'n_estimators' : n_estimator,
	'max_depth': max_depth,
}


def randomForestClassify(testX, testY, trainX, trainY, best_params):

	clf = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'], random_state=0)
	clf.fit(trainX, trainY)
	score = clf.score(testX, testY)
	# prints Gini impurity score of the random forest classifier
	print("Gini accuracy score of Forest Walk: ", score)
	# returns the predictions from test data set
	return clf.predict(testX)

#fine tuning the classifier with grid search, refit_score refers to the criterion to tune the hyperparameters to
def gridSearchWrapper(testX, testY, trainX, trainY, refit_score='accuracy_score',):
	scorers = {
		'accuracy_score': make_scorer(accuracy_score),
		'precision_score': make_scorer(precision_score),
		'recall_score': make_scorer(recall_score)
	}
	# initiates an 'empty' random forest classfier, ready to be initialized with different hyperparameters
	clf = RandomForestClassifier(n_jobs=-1)
	#Stratified sampling conducted with a minimum sample split of 10
	skf = StratifiedKFold(n_splits=10)
	grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
						   cv=skf, return_train_score=True, n_jobs=-1)
	grid_search.fit(trainX, trainY)

	# make the predictions
	y_pred = grid_search.predict(testX)
	# Prints out the optimal parameters for the criterion stated
	print('Best params for ', refit_score, ': ', grid_search.best_params_)

	# confusion matrix on the test data.
	results = pd.DataFrame(grid_search.cv_results_)
	results = results.sort_values(by='mean_test_precision_score', ascending=False)
	return results, grid_search.best_params_

# unit test code
if __name__ == '__main__':
	data = ic.separateImport()
	data = procd.fillData(data, fill_method="median")
	# in above function, fill_method has 'median', 'mode', and 'mean' options to fill data with the median, mode or mean

	testX, testY, trainX, trainY = procd.createTrainingSet(data)
	res, best_params = gridSearchWrapper(testX, testY, trainX, trainY)

	randomForestClassify(testX, testY, trainX, trainY, best_params)
	print(res)
