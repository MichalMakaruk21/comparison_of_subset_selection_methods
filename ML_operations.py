import data_operations as do
import itertools
import pandas as pd
import numpy as np
import sklearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
import math

ds = do.DataSplitter()
sub_df = do.SubDataFrameGenerator()


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

        logs_df = {"Model_criterion": [],
                   "Feature_criterion": [],
                   "Model_criterion_value": [],
                   "Selected_features": []}

        X = data_set.drop(['y'], axis=1)
        y = data_set['y']

        selected_features = []
        remaining_features = list(X.columns)

        best_model_score = np.nan

        while remaining_features:

            best_feature_score = np.nan

            X_subset_with_best_feature_added = []

            for feature in remaining_features:

                X_subset = X[selected_features + [feature]]

                model = sm.Logit(y, X_subset)
                result = model.fit(disp=False)

                if self.feature_criterion == 'p-value':
                    score = result.pvalues[feature]
                    if score > best_feature_score:
                        best_feature_score = score
                        X_subset_with_best_feature_added = X_subset
                    else:
                        X_subset = X_subset.pop(-1)

                elif self.feature_criterion == 'pseudo-R-square':
                    score = 1 - (result.llnull / result.llf)
                    if score > best_feature_score:
                        best_feature_score = score
                        X_subset_with_best_feature_added = X_subset
                    else:
                        X_subset = X_subset.pop(-1)
                else:
                    print('Best possible feature acquired')
# -----------loop exit---------
            model_with_added_feature = sm.Logit(y, X_subset_with_best_feature_added)
            result_with_added_feature = model_with_added_feature.fit(disp=False)
            # ________________________________

            log_likelihood = model_with_added_feature.loglike(result_with_added_feature.params)

            if self.model_criterion == "AIC":
                AIC = (-2 * log_likelihood) + (2 * len(X_subset_with_best_feature_added))
                if AIC < best_model_score:
                    best_model_score = AIC

                    selected_features.append(feature)
                    remaining_features.remove(feature)

                    logs_df = logs_df.append(pd.DataFrame([self.model_criterion,
                                                           self.feature_criterion,
                                                           AIC,
                                                           X_subset_with_best_feature_added]), ignore_index=True)
                    logs_df = logs_df.sort_values(by=['Model_criterion_value'], ascending=True)

                else:
                    print('Cannot select better subset')
                    break
                # ________________________________________

            elif self.model_criterion == "BIC":
                BIC = (-2 * log_likelihood) + (len(X_subset_with_best_feature_added) * math.log(len(X_subset_with_best_feature_added), math.e))
                if BIC > best_model_score:
                    best_model_score = BIC

                    selected_features.append(feature)
                    remaining_features.remove(feature)

                    logs_df = logs_df.append(pd.DataFrame([self.model_criterion,
                                                           self.feature_criterion,
                                                           BIC,
                                                           X_subset_with_best_feature_added]), ignore_index=True)
                    logs_df = logs_df.sort_values(by=['Model_criterion_value'], ascending=False)

                else:
                    print('Cannot select better subset')
                    break
                    # __________________________
            else:
               raise ValueError("Invalid stopping criterion. Correct values: 'AIC' or 'BIC'.")

            print('working')
            # else:
            # selected_features.remove(feature)
        # final_df = data_set[selected_features].assign(y=y)

        return logs_df

    def evaluate_model(self, pre_splitted=False):

        return 0


# class BackwardStepwiseSelection:

# class HybridStepwiseSelection:

class Lasso:
    @staticmethod
    def perform_lasso_logistic_regression(df: pd.DataFrame(),
                                          df_pre_split: pd.DataFrame() = None,
                                          pre_split=False):
        if pre_split:
            X_train, X_test, y_train, y_test = ds.split_data(data_set=df,
                                                             data_set_if_pre=df_pre_split,
                                                             pre_split=pre_split)
            model = LogisticRegression(penalty='l1', solver='liblinear')
            train = model.fit(X_train, y_train)

            y_predict = train.predict(X_test)
            return Metrics.return_conf_matrix_related_metrics(y_test=y_test, y_predict=y_predict)

        if not pre_split:
            X_train, X_test, y_train, y_test = ds.split_data(data_set=df, pre_split=pre_split)
            model = LogisticRegression(penalty='l1', solver='liblinear')

            train = model.fit(X_train, y_train)

            y_predict = train.predict(X_test)

            return Metrics.return_conf_matrix_related_metrics(y_test=y_test, y_predict=y_predict)


class KrossValidation:
    """
    DataSet 4 is not supprted (pre_split from vendor)
    """

    @staticmethod
    def perform_kross_validation_train(df: pd.DataFrame()):
        scoring = ['accuracy', 'precision', 'recall', 'f1']

        X = df.drop(['y'], axis=1)
        y = df['y']

        model = LogisticRegression()

        k_folds = KFold(n_splits=10)
        # coud not find additonal liblaries
        print(sklearn.metrics.get_scorer_names())
        scores = cross_validate(model, X, y, cv=k_folds, scoring=scoring, return_train_score=True)
        return {'accuracy': scores['test_accuracy'].mean(),
                'precision': scores['test_precision'].mean(),
                'recall': scores['test_recall'].mean(),
                'f1': scores['test_f1'].mean()}


class FeaturePermutation:
    @staticmethod
    def perform_selection_based_on_permutation(df: pd.DataFrame(),
                                               df_pre_split: pd.DataFrame() = None,
                                               pre_split=False):

        if pre_split:
            X_train, X_test, y_train, y_test = ds.split_data(data_set=df,
                                                             data_set_if_pre=df_pre_split,
                                                             pre_split=pre_split)
            model = LogisticRegression()
            perm_imp = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
            return

        if not pre_split:
            X_train, X_test, y_train, y_test = ds.split_data(data_set=df, pre_split=pre_split)
            df_columns = ds.return_columns(data_set=df, pre_split=False)

            model = LogisticRegression()
            train = model.fit(X_train, y_train)

            y_predict = train.predict(X_test)
            conf_matrix_metrics = Metrics().return_conf_matrix_related_metrics(y_test=y_test, y_predict=y_predict)
            print(conf_matrix_metrics)
            orginal_accuracy = conf_matrix_metrics['accuracy']
            orginal_f1 = conf_matrix_metrics['f1_score']

            X_test_with_columns = pd.DataFrame(data=X_test, columns=df_columns)

            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

            # importance_scores = result.importances_mean
            # for feature, importance_score in zip(range(X_test_with_columns.shape[1]), importance_scores):
            #     print(f"Feature {feature}: {importance_score:.4f}")

            feature_importance_df = pd.DataFrame(
                {"Feature": range(X_test_with_columns.shape[1]), "Importance": result.importances_mean}
            )

            feature_importance_df = feature_importance_df[feature_importance_df['Importance'] > 0.05]

            print(feature_importance_df)

            return "done"


class Metrics:
    @staticmethod
    def return_conf_matrix_related_metrics(y_test, y_predict):
        tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_predict).ravel()

        # sensitivity as well
        metrics = {'recall': tp / (tp + fn),
                   'precision': tp / (tp + fp),
                   'specificity': tn / (tn + fp),
                   'negative_predictive_value': tn / (tn + fn),
                   'accuracy': (tp + tn) / (tp + tn + fp + tn),
                   'f1_score': 2 * (((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn))))}
        return metrics
