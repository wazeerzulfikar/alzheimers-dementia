import glob, os, math, time, re, csv
from decimal import *
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
np.random.seed(0)
 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import scale
 
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
 
print("Loading boston")
 
data_df = load_boston()
df = pd.DataFrame(data=data_df['data'], columns=data_df['feature_names'])
X = df.values
 
print(X.shape) # (506, 13) (506,)
 
data = np.transpose(X)
# data = scale(np.transpose(X))
 
# kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
# kmeans.fit(data)
# centroids = kmeans.cluster_centers_
# print(centroids)
# predictions = kmeans.predict(data)
 
# clustering = AgglomerativeClustering(n_clusters=2, affinity='l1').fit(data)
# predictions = clustering.labels_
 
 
######### Hierarchical Clustering based on correlation
Y = pdist(data, 'correlation')
linkage = linkage(Y, 'complete')
dendrogram(linkage, color_threshold=0)
predictions = fcluster(linkage, 0.5 * Y.max(), 'distance')
plt.show()
 
print(data_df['feature_names'])
print(predictions)
print('0: {} \t 1: {}'.format(list(predictions).count(0), list(predictions).count(1)))
 
### TAX and B together for with and without scaling
 
### Hierarchical Clustering [1 2 1 3 1 2 1 2 1 1 1 2 1]
###['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO', 'B' 'LSTAT']