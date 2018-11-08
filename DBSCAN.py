import processdata as procd
import importcsv as ic
import preprocessing
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

def perform_dbscan():
	#import data table
	data = ic.separateImport()
	data = procd.fillData(data, fill_method='median')

	#several rows of cholestrol data show a value of 0, needs to be removed
	empty_indices= []
	for i in range(data.shape[0]):
		if (data['chol'][i] == 0):
			empty_indices.append(i)
	data = data.drop(data.index[empty_indices])

	#partition data into data and prediction
	X_data, Y_data = preprocessing.createFullSet(data)

	#scale the data
	X = (X_data - np.mean(X_data,axis = 0))/np.std(X_data,axis = 0)

	#perform DBSCAN with eps = 2.4 and 7 min samples
	db = DBSCAN(eps=240/100, min_samples=7).fit(X)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	#get number of clusters
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	#print cluster data
	unique, counts = np.unique(labels, return_counts=True)
	print("Number of data points in clusters")
	print(dict(zip(unique, counts)))

	#visualise data
	import visualise
	visualise.parallelVisualise(data, labels, ('black','blue','red','green'), 'dbscan')
	
	#display histogram of every cluster for every dimension
	zero_indices = np.where(labels == 0)
	one_indices = np.where(labels == 1)
	two_indices = np.where(labels == 2)

	#print histograms
	for i in range(1,10,1):
		curr_data = X_data[:,i]
		import matplotlib.pyplot as plt
		plt.figure()
		plt.hist(curr_data[zero_indices], bins='auto',color='blue')  # arguments are passed to np.histogram
		plt.title(list(data)[i] + ' cluster Zero')
		plt.savefig('./figures/'+list(data)[i] + '-cluster Zero')
		plt.figure()
		plt.hist(curr_data[one_indices], bins='auto',color='orange')  # arguments are passed to np.histogram
		plt.title(list(data)[i]+ 'cluster One')
		plt.savefig('./figures/'+list(data)[i] + '-cluster One')
		plt.figure()
		plt.hist(curr_data[two_indices], bins='auto',color='green')  # arguments are passed to np.histogram
		plt.title(list(data)[i]+ ' cluster Two')
		plt.savefig('./figures/'+list(data)[i] + '-cluster Two')
	plt.figure()
	plt.hist(Y_data[zero_indices], bins='auto',color='blue')  # arguments are passed to np.histogram
	plt.title('prediction' + ' cluster Zero')
	plt.savefig('./figures/'+'prediction cluster Zero')
	plt.figure()
	plt.hist(Y_data[one_indices], bins='auto',color='orange')  # arguments are passed to np.histogram
	plt.title('prediction' + ' cluster One')
	plt.savefig('./figures/'+'prediction cluster One')
	plt.figure()
	plt.hist(Y_data[two_indices], bins='auto',color='green') 
	plt.title('prediction' + ' cluster Two')
	plt.savefig('./figures/'+'prediction cluster Two')

