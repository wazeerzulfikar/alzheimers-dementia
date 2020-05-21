import pandas as pd
from sklearn.datasets import load_boston
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


def load_dataset(config):

    if config.dataset=='boston':
        data = _boston(config)        

    elif config.dataset=='cement':
        data = _cement(config)

    elif config.dataset=='energy':
        data = _cement(config)

    elif config.dataset=='cement':
        data = _cement(config)

    return data

def feature_split(features):
    from scipy.cluster.hierarchy import linkage
    data = np.transpose(features)
    # print(data.shape)
    ######### Hierarchical Clustering based on correlation
    Y = pdist(data, 'correlation')
    linkage = linkage(Y, 'complete')
    predictions = fcluster(linkage, 2, 'maxclust')
    # print(predictions)
    f1 = [i for i in range(len(predictions)) if predictions[i]==1]
    f2 = [i for i in range(len(predictions)) if predictions[i]==2]
    # print(data[f1].shape, data[f2].shape)
    X1 = np.transpose(data[f1])
    X2 = np.transpose(data[f2])
    # print(X1.shape, X2.shape)
    return X1, X2

def _boston(config):
    print("Loading boston")
    data_df = load_boston()
    df = pd.DataFrame(data=data_df['data'], columns=data_df['feature_names'])
    df['TARGET'] = data_df['target']

    if config.mod_split=='none':
        X = df.values
        y = df['target']
        data = {'X':X, 'y':y}

    elif config.mod_split=='human':
        features1 = ['ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX']
        features2 = ['CRIM', 'PTRATIO', 'B', 'LSTAT']
        X1 = df[features1].values
        X2 = df[features2].values
        y = df['TARGET']
        data = {'X1':X1, 'X2':X2, 'y':y}

    elif config.mod_split=='computation_split':
        X1, X2 = feature_split((df.drop(columns=['TARGET'])).values)
        y = df['TARGET']
        data = {'X1':X1, 'X2':X2, 'y':y}
    return data

def _cement(config):
    print("Loading cement")
    data=0
    return data