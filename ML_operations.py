import data_operations as data_op

import itertools
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
import statsmodels.api as sm



class DataSplitter:
    def __int__(self, data_set: pd.DataFrame(),
                      test_size=0.2,
                      random_state=21,
                      pre_split=False):
        self.data_set = data_set
        self.test_size = test_size
        self.random_state = random_state
        self.pre_split = pre_split

    def split_data(self):

        if self.data_set == 1:
            df = data_op.DataSet1()







# class BestSubsetSelection:

# class ForwardStepwiseSelection:

# class BackwardStepwiseSelection:

# class HybridStepwiseSelection:

# class Lasso:

# class KrossValidation:

# class FeaturePermutation:
