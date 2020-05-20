import glob
import os
import math
import time
import re
import csv
import pandas as pd
from sklearn.datasets import load_boston

def load_dataset(config):

    if config.dataset=='boston':
        print("Loading boston")
        data_df = load_boston()
        df = pd.DataFrame(data=data_df['data'], columns=data_df['feature_names'])
        
        X = df.values
        y = data_df['target']
        df['TARGET'] = data_df['target']
        X1 = df[['ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX']].values
        X2 = df[['CRIM', 'PTRATIO', 'B', 'LSTAT']].values
        y = df['TARGET']
        
        if config.mod_split=='none':
            data = {'X':X, 'y':y}
        elif config.mod_split=='none':
            data = {'X1':X1, 'X2':X2, 'y':y}

    return data