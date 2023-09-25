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


class ForwardStepwiseSelection:
    """
    model criteria: AIC or BIC
    feature criteria: p-value or pseudo-R-square
    """

    def __init__(self, model_criterion: str, feature_criterion=str()):
        self.model = None
        self.model_criterion = model_criterion
        self.feature_criterion = feature_criterion
        self.logs_df = pd.DataFrame({"Model_criterion": [],
                                     "Feature_criterion": [],
                                     "Model_criterion_value": [],
                                     "Selected_features": []})

    def append_log(self,
                   model_criterion: str,
                   feature_criterion: str,
                   model_criterion_value: float,
                   selected_features: list) -> pd.DataFrame():

        new_row = {"Model_criterion": model_criterion,
                   "Feature_criterion": feature_criterion,
                   "Model_criterion_value": model_criterion_value,
                   "Selected_features": selected_features}

        # self.logs_df = self.logs_df.append(new_row, ignore_index=True)
        self.logs_df = pd.concat([self.logs_df, pd.DataFrame([new_row])], ignore_index=True)
        self.logs_df = self.logs_df.sort_values(by=['Model_criterion_value'], ascending=False)
        return self.logs_df

    def select_subset(self, df: pd.DataFrame(),
                      df_pre_split: pd.DataFrame() = None,
                      pre_split: bool = None):

        # print(f'columns len: {len(ds.return_columns(data_set=df))}')

        X_train, X_test, y_train, y_test, columns = ds.split_data(data_set=df,
                                                                  data_set_if_pre=df_pre_split,
                                                                  pre_split=pre_split,
                                                                  dict_columns=True)

        init_best_feature_index = Metrics.get_best_feature_criterion_for_init_model(X_train,
                                                                                    y_train,
                                                                                    self.feature_criterion)

        columns_idx = list(range(len(ds.return_columns(data_set=df))))
        selected_features = [columns_idx[init_best_feature_index]]
        remaining_features = list(filter(lambda idx: idx != init_best_feature_index, columns_idx))

        print(f"columns indexes: {columns_idx}")


        best_model_score = Metrics.count_model_criterion(X_train,
                                                         y_train,
                                                         self.model_criterion)
        # best_model_score = math.log(2*len(remaining_features))
        # best_model_score = 2 * len(remaining_features) if self.model_criterion == "AIC" else None
        # set as log(2 * k) in AIC
        # find best start val for BIC

        while remaining_features:
            feature, score = self.feature_criterion_eval(X_train,
                                                         y_train,
                                                         selected_features,
                                                         remaining_features)

            X_subset_with_best_feature_added = X_train[selected_features + [feature]]

            X_subset_with_best_feature_added, best_model_score, selected_features, \
                remaining_features = self.model_criterion_eval(y=y_train,
                                                               X_subset_with_best_feature_added=X_subset_with_best_feature_added,
                                                               best_model_score=best_model_score,
                                                               selected_features=selected_features,
                                                               remaining_features=remaining_features,
                                                               feature=feature)

            print()
            print(selected_features)
            print(best_model_score)

            # best_feature_score = 0 if self.feature_criterion == "pseudo-R-square" else None
            """
            X_subset_with_best_feature_added = []

            for feature in remaining_features:

                X_subset = X_train[selected_features + [feature]]

                try:
                    model = sm.Logit(y_train, X_subset)
                    result = model.fit(disp=False)
                except Exception as e:
                    print(f"Error occurred while training initial model: {e}")
                    remaining_features.remove(feature)
                    continue
                else:
                    if self.feature_criterion == 'p-value':
                        score = result.pvalues[feature]

                        if best_feature_score is None:
                            best_feature_score = score
                            X_subset_with_best_feature_added = X_subset

                            X_subset_with_best_feature_added, best_model_score, selected_features, \
                            remaining_features = self.model_criterion_eval(y=y_train,
                                                                           X_subset_with_best_feature_added=X_subset_with_best_feature_added,
                                                                           best_model_score=best_model_score,
                                                                           selected_features=selected_features,
                                                                           remaining_features=remaining_features,
                                                                           feature=feature)

                        elif score < best_feature_score:
                            best_feature_score = score
                            X_subset_with_best_feature_added = X_subset

                            X_subset_with_best_feature_added, best_model_score, selected_features, \
                            remaining_features = self.model_criterion_eval(y=y_train,
                                                                           X_subset_with_best_feature_added=X_subset_with_best_feature_added,
                                                                           best_model_score=best_model_score,
                                                                           selected_features=selected_features,
                                                                           remaining_features=remaining_features,
                                                                           feature=feature)
                        else:
                            X_subset = X_subset[:-1]

                    elif self.feature_criterion == 'pseudo-R-square':

                        score = 1 - (result.llnull / result.llf)

                        if score > best_feature_score:
                            best_feature_score = score
                            X_subset_with_best_feature_added = X_subset

                            X_subset_with_best_feature_added, best_model_score, selected_features, \
                            remaining_features = self.model_criterion_eval(y=y_train,
                                                                           X_subset_with_best_feature_added=X_subset_with_best_feature_added,
                                                                           best_model_score=best_model_score,
                                                                           selected_features=selected_features,
                                                                           remaining_features=remaining_features,
                                                                           feature=feature)
                        else:
                            X_subset = X_subset[:-1]
                    else:
                        print('Set correct feature criterion parameter')
                    # -----------loop exit-----------------------------------------

            print('working')
            """
        return self.logs_df

    def feature_criterion_eval(self,
                               X_train: pd.DataFrame(),
                               y_train: pd.DataFrame(),
                               selected_features: list,
                               remaining_features: list):

        scores_dict = pd.DataFrame({"columns": [], "scores": []})
        id_min = None

        for feature in remaining_features:

            X_subset = X_train[selected_features + [feature]]

            model = sm.Logit(y_train, X_subset)
            result = model.fit(disp=False)

            if self.feature_criterion == "p-value":

                scores_dict['columns'].append(feature)
                scores_dict['scores'].append(result.pvalues[feature])

            # __________________________________________#

            elif self.feature_criterion == "pseudo_R_square":
                score = 1 - (result.llnull / result.llf)
                # TO DOOOOOOOO
                id_min = "xdd"
                return 0
            id_min = scores_dict[['scores']].idmin(axis=0) if self.feature_criterion == "p-value" else "Pseudo_RSQUARE"

        return scores_dict['columns'][id_min], scores_dict['scores'][id_min]

    def model_criterion_eval(self,
                             y,
                             X_subset_with_best_feature_added,
                             best_model_score,
                             selected_features,
                             remaining_features,
                             feature,
                             ):

        model_with_added_feature = sm.Logit(y, X_subset_with_best_feature_added)
        result_with_added_feature = model_with_added_feature.fit(disp=False)
        # ________________________________

        log_likelihood = model_with_added_feature.loglike(result_with_added_feature.params)

        if self.model_criterion == "AIC":

            # set star AIC val as 10

            AIC = (-2 * log_likelihood) + (2 * len(X_subset_with_best_feature_added))

            # assign first value to model score

            if AIC < best_model_score:
                best_model_score = AIC

                selected_features.append(feature)
                remaining_features.remove(feature)

                self.append_log(self.model_criterion,
                                self.feature_criterion,
                                AIC,
                                selected_features)
            else:
                print('Cannot select better subset AIC')
                remaining_features.remove(feature)

        elif self.model_criterion == "BIC":
            # check if BIC calculates correctly
            BIC = (-2 * log_likelihood) + (
                    len(X_subset_with_best_feature_added) * math.log(len(X_subset_with_best_feature_added)))
            if best_model_score is None:
                best_model_score = BIC

                selected_features.append(feature)
                remaining_features.remove(feature)

                self.append_log(self.model_criterion,
                                self.feature_criterion,
                                BIC,
                                selected_features)

            elif BIC > best_model_score:
                best_model_score = BIC

                selected_features.append(feature)
                remaining_features.remove(feature)

                self.append_log(self.model_criterion,
                                self.feature_criterion,
                                BIC,
                                selected_features)

            else:
                print('Cannot select better subset BIC')
                remaining_features.remove(feature)
                # __________________________
        else:
            raise ValueError("Invalid stopping criterion. Correct values: 'AIC' or 'BIC'.")

        return X_subset_with_best_feature_added, best_model_score, selected_features, remaining_features


    def x_set_selector(self, X_set=):

    def evaluate_model(self, pre_splitted=False):

        return 0


class BackwardStepwiseSelection:
    """
        model criteria: AIC or BIC
        feature criteria: p-value or pseudo-R-square
        """

    def __init__(self, model_criterion: str, feature_criterion=str()):
        self.model = None
        self.model_criterion = model_criterion
        self.feature_criterion = feature_criterion
        self.logs_df = pd.DataFrame({"Model_criterion": [],
                                     "Feature_criterion": [],
                                     "Model_criterion_value": [],
                                     "Selected_features": []})

    def append_log(self,
                   model_criterion: str,
                   feature_criterion: str,
                   model_criterion_value: float,
                   selected_features: list) -> pd.DataFrame():

        new_row = {"Model_criterion": model_criterion,
                   "Feature_criterion": feature_criterion,
                   "Model_criterion_value": model_criterion_value,
                   "Selected_features": selected_features}

        # self.logs_df = self.logs_df.append(new_row, ignore_index=True)
        self.logs_df = pd.concat([self.logs_df, pd.DataFrame([new_row])], ignore_index=True)
        self.logs_df = self.logs_df.sort_values(by=['Model_criterion_value'], ascending=False)
        return self.logs_df

    def select_subset(self, data_set: pd.DataFrame()):

        X = data_set.drop(['y'], axis=1)
        y = data_set['y']

        selected_features = []
        remaining_features = list(X.columns)

        # train initial model with all features
        try:
            init_model = sm.Logit(y, X)
            init_result = init_model.fit(disp=False)

            init_log_likelihood = init_model.loglike(init_result.params)

            initial_model_score = (-2 * init_log_likelihood) + (
                    2 * len(X.columns)) if self.model_criterion == 'AIC' else (-2 * init_log_likelihood) + (
                    len(X.columns) * math.log(len(X.columns), math.e))

            self.append_log(self.model_criterion,
                            self.feature_criterion,
                            initial_model_score,
                            X.columns)

            best_model_score = initial_model_score

        except Exception as e:
            print(f"Error occurred while training initial model: {e}")
            return None

        else:

            while remaining_features:

                worst_feature_score = None

                X_subset_with_worst_feature_dropped = []

                for feature in remaining_features:
                    # remove returns null list
                    # ll = [x for x in selected_features if x != feature]
                    X_subset = X[[x for x in remaining_features if x != feature]]
                    display(f"y variable:\n {y}")
                    display(f"X variables:\n {X_subset}")

                    try:
                        model = sm.Logit(y, X_subset)
                        result = model.fit(disp=False)
                    except Exception as e:
                        print(f"Error occurred while training initial model: {e}")
                        selected_features.remove(feature)
                        remaining_features.remove(feature)
                        continue
                    else:

                        if self.feature_criterion == 'p-value':
                            score = result.pvalues[feature]

                            if worst_feature_score is None:
                                worst_feature_score = score
                                X_subset_with_worst_feature_dropped = X_subset

                                X_subset_with_worst_feature_dropped, best_model_score, selected_features, \
                                    remaining_features = self.model_criterion_eval(y=y,
                                                                                   X_subset_with_worst_feature_dropped=X_subset_with_worst_feature_dropped,
                                                                                   best_model_score=best_model_score,
                                                                                   selected_features=selected_features,
                                                                                   remaining_features=remaining_features,
                                                                                   feature=feature)

                            elif score < worst_feature_score:
                                worst_feature_score = score
                                X_subset_with_worst_feature_dropped = X_subset

                                X_subset_with_worst_feature_dropped, best_model_score, selected_features, \
                                    remaining_features = self.model_criterion_eval(y=y,
                                                                                   X_subset_with_worst_feature_dropped=X_subset_with_worst_feature_dropped,
                                                                                   best_model_score=best_model_score,
                                                                                   selected_features=selected_features,
                                                                                   remaining_features=remaining_features,
                                                                                   feature=feature)
                            else:
                                X_subset = X[selected_features + [feature]]
                        elif self.feature_criterion == 'pseudo-R-square':
                            score = 1 - (result.llnull / result.llf)

                            if worst_feature_score is None:
                                worst_feature_score = score
                                X_subset_with_worst_feature_dropped = X_subset

                                X_subset_with_worst_feature_dropped, best_model_score, selected_features, \
                                    remaining_features = self.model_criterion_eval(y=y,
                                                                                   X_subset_with_worst_feature_dropped=X_subset_with_worst_feature_dropped,
                                                                                   best_model_score=best_model_score,
                                                                                   selected_features=selected_features,
                                                                                   remaining_features=remaining_features,
                                                                                   feature=feature)

                            elif score > worst_feature_score:
                                worst_feature_score = score
                                X_subset_with_worst_feature_dropped = X_subset

                                X_subset_with_worst_feature_dropped, best_model_score, selected_features, \
                                    remaining_features = self.model_criterion_eval(y=y,
                                                                                   X_subset_with_worst_feature_dropped=X_subset_with_worst_feature_dropped,
                                                                                   best_model_score=best_model_score,
                                                                                   selected_features=selected_features,
                                                                                   remaining_features=remaining_features,
                                                                                   feature=feature)
                            else:
                                X_subset = X[selected_features + [feature]]
                        else:
                            print('Set correct feature criterion parameter')
                        # -----------loop exit-----------------------------------------

            print('working')

        return self.logs_df

    def model_criterion_eval(self,
                             y,
                             X_subset_with_worst_feature_dropped,
                             best_model_score,
                             selected_features,
                             remaining_features,
                             feature,
                             ):

        model_with_dropped_feature = sm.Logit(y, X_subset_with_worst_feature_dropped)
        result_with_dropped_feature = model_with_dropped_feature.fit(disp=False)
        # ________________________________

        log_likelihood = model_with_dropped_feature.loglike(result_with_dropped_feature.params)

        if self.model_criterion == "AIC":
            AIC = (-2 * log_likelihood) + (2 * len(X_subset_with_worst_feature_dropped))

            if AIC < best_model_score:
                best_model_score = AIC

                selected_features.append(feature)
                remaining_features.remove(feature)

                self.append_log(self.model_criterion,
                                self.feature_criterion,
                                AIC,
                                selected_features)
            else:
                print('Cannot select better subset AIC')
                remaining_features.remove(feature)

        elif self.model_criterion == "BIC":
            BIC = (-2 * log_likelihood) + (
                    len(X_subset_with_worst_feature_dropped) * math.log(len(X_subset_with_worst_feature_dropped),
                                                                        math.e))
            if BIC > best_model_score:
                best_model_score = BIC

                selected_features.append(feature)
                remaining_features.remove(feature)

                self.append_log(self.model_criterion,
                                self.feature_criterion,
                                BIC,
                                selected_features)

            else:
                print('Cannot select better subset BIC')
                remaining_features.remove(feature)
                # __________________________
        else:
            raise ValueError("Invalid stopping criterion. Correct values: 'AIC' or 'BIC'.")

        return X_subset_with_worst_feature_dropped, best_model_score, selected_features, remaining_features

    def evaluate_model(self, pre_splitted=False):

        return 0


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


class CrossValidation:
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


class BruteForce:
    """
    feature criterion should be "pseudo_R_square" or "p_vlaue"
    feed model only with variables which match feature criterion
    Data set 4 cannot run due to lack of memeory
    """

    def __init__(self, feature_criterion: str, criterion_val=float):
        self.feature_criterion = feature_criterion
        self.criterion_val = criterion_val

    def select_subset(self,
                      df: pd.DataFrame(),
                      df_pre_split: pd.DataFrame() = None,
                      pre_split=False
                      ):

        X_train, X_test, y_train, y_test = ds.split_data(data_set=df,
                                                         data_set_if_pre=df_pre_split,
                                                         pre_split=pre_split)

        model = sm.Logit(y_train, X_train)
        result = model.fit(disp=False)

        if self.feature_criterion == "p_value":
            selected_features_idx = [idx for idx, feature in enumerate(result.pvalues) if feature <= float(self.criterion_val)]

            X_train = X_train[:, selected_features_idx]
            X_test = X_test[:, selected_features_idx]
            return X_train, X_test, y_train, y_test

        if self.feature_criterion == "pseudo_R_square":
            print(f"p_val: {result.pvalues}")
            print(f"params: {result.params}")

            params = result.params

            log_like = model.loglikeobs(params=params)
            log_like_fit = result.llnull


            print("______________________")

            print(len(log_like))
            print(log_like_fit)

            # xd = (log_like - log_like_fit) / log_like

            xd =[(x - log_like_fit) for x in log_like]

            print(f"idk R_square_list: {xd}")
            # for x in xd:
                # print(x)

            score = 1 - (result.llnull / result.llf)
            print(score)
            # xd = model.pseudo_rsquared(kind=' “mcf”')


            return 0

    def eval_model(self):

        return 0


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

    @staticmethod
    def get_best_feature_criterion_for_init_model(X_train: list,
                                                  y_train: list,
                                                  feature_criterion: str, ):

        scores_dict = pd.DataFrame({"columns": [], "scores": []})

        model = sm.Logit(y_train, X_train)
        result = model.fit(disp=False)

        if feature_criterion == "p-value":
            # print(f'results pvales len: {len(result.pvalues)}')
            min_id = np.argmin(result.pvalues)



            return np.argmin(result.pvalues)

        elif feature_criterion == "pseudo_R_square":
            score = 1 - (result.llnull / result.llf)
            # TO DOOOOOOOO

            return 0

    @staticmethod
    def count_model_criterion(X_train: pd.DataFrame(),
                              y_train: pd.DataFrame(),
                              model_criterion: str) -> float():

        model = sm.Logit(y_train, X_train)
        result = model.fit(disp=False)

        log_likelihood = model.loglike(result.params)

        if model_criterion == "AIC":
            return (-2 * log_likelihood) + (2 * len(X_train))

        elif model_criterion == 'BIC':
            return (-2 * log_likelihood) + (len(X_train) * math.log(len(X_train)))

class StepwiseSelection:

    def __init__(self):
        pass

