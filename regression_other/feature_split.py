import glob, os, math, time, re, csv
from decimal import *
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
np.random.seed(0)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

print("Loading boston")

data_df = load_boston()
df = pd.DataFrame(data=data_df['data'], columns=data_df['feature_names'])
X = df.values

print(X.shape) # (506, 13) (506,)

data = np.transpose(X)
# data = scale(np.transpose(X))
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
kmeans.fit(data)
centroids = kmeans.cluster_centers_
print(centroids)
predictions = kmeans.predict(data)
print(data_df['feature_names'])
print(predictions)
print('0: {} \t 1: {}'.format(list(predictions).count(0), list(predictions).count(1)))

### TAX and B together for with and without scaling