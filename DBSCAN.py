import processdata as procd
import importcsv as ic
import preprocessing
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

if __name__=='__main__':
    data = ic.separateImport()
    data = procd.fillData(data, fill_method='median')
    X_data, Y_data = preprocessing.createFullSet(data)
    # print(X_data)
    # print(Y_data)
    # X = StandardScaler().fit_transform(X_data)
    X = (X_data - np.mean(X_data))/np.std(X_data)
    db = DBSCAN(eps=45/100, min_samples=5).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    import visualise

    count1 = 0
    count0 = 0
    oneandone = 0
    for i, x in enumerate(data['fbs']):
        if x == 1.0:
            if labels[i] == 1:
                count1+=1
            else:
                oneandone +=1
        else:
            count0+=1
    print(count1, count0, oneandone)
    print(labels)
    visualise.parallelVisualise(data, labels, ('red','green', 'blue'), 'dbscan')
	# print (X)

	# #############################################################################
	# Compute DBSCAN
	# max_score = 0
	# sample_num = 0
	# eps_num = 0
	# for samples in range(1,50,2):
	# 	for eps in range (5, 50, 5):

    		# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    		# print(max_score, sample_num, eps_num)
    		# if n_clusters_ <= 1:
    		# 	continue
    		# if (max_score < metrics.silhouette_score(X, labels)):
    		# 	max_score = metrics.silhouette_score(X, labels)
    		# 	sample_num = samples
    		# 	eps_num = eps


    # Number of clusters in labels, ignoring noise if present.

    # unique, counts = np.unique(labels, return_counts=True)
	# print(dict(zip(unique, counts)))
	# print(Y_data)

	# print('Estimated number of clusters: %d' % n_clusters_)
	# print("Homogeneity: %0.3f" % metrics.homogeneity_score(Y_data, labels))
	# print("Completeness: %0.3f" % metrics.completeness_score(Y_data, labels))
	# print("V-measure: %0.3f" % metrics.v_measure_score(Y_data, labels))
	# print("Adjusted Mutual Information: %0.3f"
	# 	  % metrics.adjusted_mutual_info_score(Y_data, labels))
	# print("Silhouette Coefficient: %0.3f"
	# 	  % metrics.silhouette_score(X, labels))

	# b = np.where(labels == 0)
	# c = X_data[b]
	# d = Y_data[b]
	# print(c.shape, d.shape)
	# e = np.concatenate([c,d],axis = 1)
	# print(c.shape)
	# print(c)
	# print(e.shape)
	# print(e)

	# # #############################################################################
	# # Plot result
	# import matplotlib.pyplot as plt

	# # Black removed and is used for noise instead.
	# unique_labels = set(labels)
	# colors = [plt.cm.Spectral(each)
	#           for each in np.linspace(0, 1, len(unique_labels))]
	# for k, col in zip(unique_labels, colors):
	#     if k == -1:
	#         # Black used for noise.
	#         col = [0, 0, 0, 1]

	#     class_member_mask = (labels == k)

	#     xy = X[class_member_mask & core_samples_mask]
	#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
	#              markeredgecolor='k', markersize=14)

	#     xy = X[class_member_mask & ~core_samples_mask]
	#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
	#              markeredgecolor='k', markersize=6)

	# plt.title('Estimated number of clusters: %d' % n_clusters_)
	# plt.show()
