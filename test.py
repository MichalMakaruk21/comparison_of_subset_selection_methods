import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
import numpy as np
from IPython.display import display

# data = pd.read_csv('data/Polish_companies_bankruptcy/polish_companies_bankruptcy_3.csv', sep=',', index_col='id', low_memory=False)
"""
for x in data.columns:
    s = f'data.{x}.dtype'
    try:
        print(exec(s))
    except Exception:
        print(f'{s} has error')
"""

# display(len(data.columns))
import data_operations as do

ds1 = do.Dataset2()

x, y = ds1.preprocess_data(scaler_type="StandardScaler")

print(x)