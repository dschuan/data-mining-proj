import pandas as pd

import processdata as procd
import importcsv as ic
from sklearn.cluster import KMeans
import preprocessing
from sklearn.model_selection import GridSearchCV
from processResults import processResults
import matplotlib.pyplot as plt

if __name__ == '__main__':
	data = ic.separateImport()
	data = procd.fillData(data, fill_method='median')
	X_data, Y_data = preprocessing.createFullSet(data)

	#Finding the optimal number of clusters using the Elbow Graph
	wcss = []
	for i in range(2, 10):
		km = KMeans(n_clusters=i, init='k-means++')
		km.fit(X_data)
		wcss.append(km.inertia_)
	plt.plot(range(2,10), wcss)
	plt.title('The Elbow Method')
	plt.xlabel('n_clusters')
	plt.ylabel('Average Within-Cluster distance to Centroid (WCSS)')
	#plt.show() #shows the Elbow Graph
	print("Best number of clusters = 3\n")

	#run K Means Clustering with no. of clusters = 3
	finalKM = KMeans(n_clusters=3, init='k-means++')
	finalKM.fit(X_data)
	dictionary = {
	'0': [],
	'1': [],
	'2': []
	}
	clusters = finalKM.labels_.tolist()
	for i in range(len(clusters)):
		x = clusters[i]
		if x == 0:
			y = Y_data[i]
			dictionary['0'].append(y)
		elif x == 1:
			y = Y_data[i]

			dictionary['1'].append(y)
		elif x == 2:
			y = Y_data[i]
			dictionary['2'].append(y)

	print("No. of 0s in Cluster 0 is " + str(dictionary['0'].count(0)))
	print("No. of 1s in Cluster 0 is " + str(dictionary['0'].count(1))+"\n")

	print("No. of 0s in Cluster 1 is " + str(dictionary['1'].count(0)))
	print("No. of 1s in Cluster 1 is " + str(dictionary['1'].count(1))+"\n")

	print("No. of 0s in Cluster 2 is " + str(dictionary['2'].count(0)))
	print("No. of 1s in Cluster 2 is " + str(dictionary['2'].count(1))+"\n")

	centroids = finalKM.cluster_centers_.tolist()
	print("Coordinates of the Centroids:\n")
	for centroid in centroids:
		print(centroids)

