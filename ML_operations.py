import data_operations as do
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


# class BestSubsetSelection:

class ForwardStepwiseSelection:
    """
    model criteria: AIC or BIC
    feature criteria: p-value or pseudo-R-square
    """

    def __init__(self, model_criterion: str, feature_criterion=str()):
        self.model = None
        self.model_criterion = model_criterion
        self.feature_criterion = feature_criterion

    def select_subset(self, data_set: pd.DataFrame()):
        X = data_set.drop(['y'], axis=1)
        y = data_set['y']

        selected_features = []
        remaining_features = list(X.columns)

        while remaining_features:
            best_model_score = float()

            for feature in remaining_features:
                # X_subset = X[selected_features.append(feature)]
                X_subset = X[selected_features + [feature]]

                model = sm.Logit(y, X_subset)
                result = model.fit(disp=False)

                if self.feature_criterion == 'p-value':
                    score = result.pvalues[feature]
                elif self.feature_criterion == 'pseudo-R-square':
                    # score = 1 - (result.llnull / result.llf)
                    score = 0
                    # find liblary to handla this
                else:
                    raise ValueError("Invalid feature criterion. Correct values: 'p-value' or 'pseudo-R-square'.")


                if self.model_criterion == "AIC":

                elif self.model_criterion == "BIC":

                else:
                    raise ValueError("Invalid stopping criterion. Correct values: 'AIC' or 'BIC'.")



                if self.model_criterion == 'AIC' and score < best_model_score:
                    best_model_score = score

                    selected_features.append(feature)
                    remaining_features.remove(feature)

                elif self.model_criterion == 'BIC' and score > best_model_score:
                    best_model_score = score

                    selected_features.append(feature)
                    remaining_features.remove(feature)
            print('working')
                # else:
                    # selected_features.remove(feature)
        final_df = data_set[selected_features].assign(y=y)
        return final_df

# class BackwardStepwiseSelection:

# class HybridStepwiseSelection:

# class Lasso:

# class KrossValidation:

# class FeaturePermutation:
