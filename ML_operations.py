import data_operations as do
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
import math

ds = do.DataSplitter()
sub_df = do.SubDataFrameGenerator()


class StepwiseSelection(object):
    def __init__(self,
                 model_criterion: str,
                 feature_criterion=str()):
        self.model = None
        self.model_criterion = model_criterion
        self.feature_criterion = feature_criterion
        self.logs_df = pd.DataFrame({"Model_criterion": [],
                                     "Model_criterion_value": [],
                                     "Selected_features": []})

    def get_feature_criterion_for_init_model(self,
                                             X_train: list,
                                             y_train: list,
                                             subset_selection: str):

        scores_dict = pd.DataFrame({"columns": [], "scores": []})

        columns_idx = [x for x in range(len(X_train[0]))]

        for feature in columns_idx:
            X_subset = self.select_based_on_idx(x_set=X_train, feature_list=[feature])

            model = sm.Logit(y_train, X_subset)
            result = model.fit(disp=False)

            new_row = {"columns": feature, "scores": result.pvalues[-1]}

            new_row_df = pd.DataFrame([new_row])

            scores_dict = pd.concat([scores_dict, new_row_df], ignore_index=True)

        # add statement for backward
        return scores_dict['columns'][scores_dict['scores'].idxmin(axis=0)] if \
            subset_selection == "forward" else scores_dict['columns'][scores_dict['scores'].idxmax(axis=0)]

    def count_model_criterion(self,
                              X_train: list,
                              y_train: list,
                              selected_features: list) -> float():
        """
        model criterion:
        AIC,
        BIC,
        pseudo_R_square
        """

        # print(f"X_train set: {X_train}")
        # print(len(X_train[0]))
        X_train = self.select_based_on_idx(X_train,
                                           selected_features)

        model = sm.Logit(y_train, X_train)
        result = model.fit(disp=False)

        if self.model_criterion == "AIC":
            return result.aic

        elif self.model_criterion == 'BIC':
            return result.bic

        elif self.model_criterion == "pseudo_R_square":
            # 1 - (result.llnull / result.llf)
            return (result.llf - result.llnull) / result.llf

    def select_based_on_idx(self,
                            x_set: np.array,
                            feature_list: list):
        feature_list = list(map(int, feature_list))
        # print(f"feature list: {feature_list}")
        cut_array = x_set[:, feature_list]
        # print(f"X_train, selected using indexes: {cut_array}")
        return cut_array

    def append_log(self,
                   model_criterion: str,
                   model_criterion_value: float,
                   selected_features: list) -> pd.DataFrame():

        new_row = {"Model_criterion": model_criterion,
                   "Model_criterion_value": model_criterion_value,
                   "Selected_features": selected_features}

        # self.logs_df = self.logs_df.append(new_row, ignore_index=True)
        self.logs_df = pd.concat([self.logs_df, pd.DataFrame([new_row])], ignore_index=True)
        self.logs_df = self.logs_df.sort_values(by=['Model_criterion_value'], ascending=False)
        return self.logs_df


class ForwardStepwiseSelection(StepwiseSelection):
    """
    model criteria: AIC, BIC, pseudo_R_square
    feature criteria: p-value
    """

    def __init__(self, model_criterion: str, feature_criterion: str = 'p_value'):
        super().__init__(model_criterion, feature_criterion)
        self.model = None
        self.model_criterion = model_criterion
        self.feature_criterion = feature_criterion

    def select_subset(self,
                      df: pd.DataFrame(),
                      df_pre_split: pd.DataFrame() = None,
                      pre_split: bool = None):

        # print(f'columns len: {len(ds.return_columns(data_set=df))}')

        X_train, X_test, y_train, y_test, columns = ds.split_data(data_set=df,
                                                                  data_set_if_pre=df_pre_split,
                                                                  pre_split=pre_split,
                                                                  dict_columns=True)

        init_best_feature_index = super().get_feature_criterion_for_init_model(X_train,
                                                                               y_train,
                                                                               subset_selection="forward")
        # print(f"init variable: {init_best_feature_index}")

        columns_idx = list(range(len(columns)))
        selected_features = [columns_idx[int(init_best_feature_index)]]
        remaining_features = list(filter(lambda idx: idx != init_best_feature_index, columns_idx))

        # print(f"columns indexes: {columns_idx}")

        best_model_score = super().count_model_criterion(X_train,
                                                         y_train,
                                                         selected_features)
        # print(f"init best model score: {best_model_score}")

        while remaining_features:
            # print(f"selected features: {selected_features}")
            # print(f"remaining features: {remaining_features}")

            feature, score = self.feature_criterion_eval(X_train,
                                                         y_train,
                                                         selected_features,
                                                         remaining_features)

            #  X_subset_with_best_feature_added = X_train[selected_features + [int(feature)]]

            best_model_score, selected_features, \
                remaining_features = self.model_criterion_eval(y=y_train,
                                                               X_train=X_train,
                                                               best_model_score=best_model_score,
                                                               selected_features=selected_features,
                                                               remaining_features=remaining_features,
                                                               feature=feature)

            # print(selected_features)
            # print(best_model_score)

            # print(f"selected_features: {selected_features}")
        # return X_train, X_test, y_train, y_test
        return X_train, X_test, y_train, y_test, self.logs_df["Selected_features"].loc[
            self.logs_df["Model_criterion_value"].idxmax() if self.logs_df["Model_criterion"].iloc[
                                                                  0] == "pseudo_R_square"
            else self.logs_df["Model_criterion_value"].idxmin()
        ]

    def feature_criterion_eval(self,
                               X_train: pd.DataFrame(),
                               y_train: pd.DataFrame(),
                               selected_features: list,
                               remaining_features: list):

        scores_dict = pd.DataFrame({"columns": [], "scores": []})

        for feature in remaining_features:
            # print(f"feature id: {feature}")
            # print(f"remaining_features: {remaining_features}")

            X_subset = super().select_based_on_idx(x_set=X_train, feature_list=selected_features + [feature])

            # print(f"X_train feature criterion eval: {X_subset}")
            # print(f"len of x_subset in feature criterion: {len(X_subset[0])}")
            # print(f"selected features: {selected_features+[feature]}")

            try:
                model = sm.Logit(y_train, X_subset)
                result = model.fit(disp=False)
            except Exception:
                temp_df = pd.DataFrame(data=X_subset, columns=selected_features + [int(feature)])
                print(temp_df.describe())
                print([temp_df.iloc[i, j] for i, j in zip(*np.where(pd.isnull(temp_df)))])

            else:

                new_row = {"columns": feature, "scores": result.pvalues[-1]}

                # print(f"feature p_val:  {result.pvalues[-1]}")

                new_row_df = pd.DataFrame([new_row])

                scores_dict = pd.concat([scores_dict, new_row_df], ignore_index=True)

        # pd.set_option('display.float_format', '{:.2f}'.format)

        id_min = scores_dict['scores'].idxmin(axis=0)
        # print(scores_dict)
        # print(id_min)
        return scores_dict['columns'][id_min], scores_dict['scores'][id_min]

    def model_criterion_eval(self,
                             y,
                             X_train,
                             best_model_score,
                             selected_features,
                             remaining_features,
                             feature,
                             ):

        test_feature_set = selected_features + [int(feature)]

        # (f"mc features set: {test_feature_set}")

        model_criterion_val = super().count_model_criterion(X_train,
                                                            y,
                                                            test_feature_set)
        # (f"model criterion: {model_criterion_val}")

        if self.model_criterion == "AIC":

            if model_criterion_val < best_model_score:
                best_model_score = model_criterion_val

                selected_features = test_feature_set
                remaining_features.remove(feature)

                super().append_log(self.model_criterion,
                                   model_criterion_val,
                                   selected_features)
            else:
                print('Cannot select better subset using AIC')
                remaining_features.remove(feature)

        elif self.model_criterion == "BIC":

            if model_criterion_val > best_model_score:
                best_model_score = model_criterion_val

                selected_features = test_feature_set
                remaining_features.remove(feature)

                super().append_log(self.model_criterion,
                                   model_criterion_val,
                                   selected_features)

            else:
                print('Cannot select better subset using BIC')
                remaining_features.remove(feature)
                # __________________________

        elif self.model_criterion == "pseudo_R_square":

            if model_criterion_val > best_model_score:
                best_model_score = model_criterion_val

                selected_features = test_feature_set
                remaining_features.remove(feature)

                super().append_log(self.model_criterion,
                                   model_criterion_val,
                                   selected_features)
            else:
                print('Cannot select better subset using pseudo R-square')
                remaining_features.remove(feature)

        else:
            raise ValueError("Invalid stopping criterion. Correct values: 'AIC', 'BIC' or pseudo_R_square.")

        return best_model_score, selected_features, remaining_features

    def evaluate_model(self,
                       df: pd.DataFrame(),
                       df_pre_split: pd.DataFrame() = None,
                       pre_split: bool = None):

        # added best subset as new value
        X_train, X_test, y_train, y_test, best_subset = self.select_subset(df=df,
                                                                           df_pre_split=df_pre_split,
                                                                           pre_split=pre_split)
        """
        best_subset = self.logs_df["Selected_features"].loc[
            self.logs_df["Model_criterion_value"].idxmax() if self.logs_df["Model_criterion"].iloc[
                                                                  0] == "pseudo_R_square"
            else self.logs_df["Model_criterion_value"].idxmin()
        ]
        """
        print(f"best_subset: {best_subset}")

        X_train = super().select_based_on_idx(X_train, best_subset)
        X_test = super().select_based_on_idx(X_test, best_subset)

        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)

        metrics_calculator = SelectedMetrics()
        return metrics_calculator.return_conf_matrix_related_metrics(y_true=y_test, y_predict=y_predict)


class BackwardStepwiseSelection(StepwiseSelection):
    """
    model criteria: AIC, BIC, pseudo_R_square
    feature criteria: p-value
    """

    def __init__(self, model_criterion: str, feature_criterion: str = 'p_value'):
        super().__init__(model_criterion, feature_criterion)
        self.model = None
        self.model_criterion = model_criterion
        self.feature_criterion = feature_criterion

    def select_subset(self,
                      df: pd.DataFrame(),
                      df_pre_split: pd.DataFrame() = None,
                      pre_split: bool = None):

        # print(f'columns len: {len(ds.return_columns(data_set=df))}')

        X_train, X_test, y_train, y_test, columns = ds.split_data(data_set=df,
                                                                  data_set_if_pre=df_pre_split,
                                                                  pre_split=pre_split,
                                                                  dict_columns=True)

        init_worst_feature_index = super().get_feature_criterion_for_init_model(X_train,
                                                                                y_train,
                                                                                subset_selection="backward")

        # print(f"init variable: {init_best_feature_index}")

        columns_idx = list(range(len(columns)))
        selected_features = list(filter(lambda idx: idx != init_worst_feature_index, columns_idx))
        remaining_features = selected_features
        not_dropped_features = []
        # print(f"columns indexes: {columns_idx}")

        best_model_score = super().count_model_criterion(X_train,
                                                         y_train,
                                                         remaining_features)
        # print(f"init best model score: {best_model_score}")

        while remaining_features:
            # print(f"selected features: {selected_features}")
            # print(f"remaining features: {remaining_features}")

            feature, score = self.feature_criterion_eval(X_train,
                                                         y_train,
                                                         selected_features,
                                                         not_dropped_features)

            #  X_subset_with_best_feature_added = X_train[selected_features + [int(feature)]]

            best_model_score, selected_features, remaining_features, \
                not_dropped_features = self.model_criterion_eval(y=y_train,
                                                                 X_train=X_train,
                                                                 best_model_score=best_model_score,
                                                                 selected_features=selected_features,
                                                                 remaining_features=remaining_features,
                                                                 not_dropped_features=not_dropped_features,
                                                                 feature=feature)

            # print(selected_features)
            # print(best_model_score)

            # print(f"selected_features: {selected_features}")
        # return X_train, X_test, y_train, y_test
        return X_train, X_test, y_train, y_test, self.logs_df["Selected_features"].loc[
            self.logs_df["Model_criterion_value"].idxmax() if self.logs_df["Model_criterion"].iloc[
                                                                  0] == "pseudo_R_square"
            else self.logs_df["Model_criterion_value"].idxmin()
        ]

    def feature_criterion_eval(self,
                               X_train: pd.DataFrame(),
                               y_train: pd.DataFrame(),
                               selected_features: list,
                               not_dropped_features: list):
        # Wrong concept somehow

        scores_dict = pd.DataFrame({"columns": [], "scores": []})

        # some features are selected because of low p_val, but model criterion already checked that dropping
        # these features have no or negative effects on model
        filtered_features = list(filter(lambda idx: idx not in not_dropped_features, selected_features))
        print(len(filtered_features))
        for feature in filtered_features:
            # print(f"feature id: {feature}")
            # print(f"remaining_features: {remaining_features}")

            feature_list = list(filter(lambda idx: idx != feature, selected_features))
            X_subset = super().select_based_on_idx(x_set=X_train, feature_list=feature_list)

            # print(f"X_train feature criterion eval: {X_subset}")
            # print(f"len of x_subset in feature criterion: {len(X_subset[0])}")
            # print(f"feature list: {feature_list}")
            try:
                model = sm.Logit(y_train, X_subset)
                result = model.fit(disp=False)
            except Exception:

                temp_df = pd.DataFrame(data=X_subset, columns=feature_list)
                print(temp_df.describe())
                print([temp_df.iloc[i, j] for i, j in zip(*np.where(pd.isnull(temp_df)))])

            else:

                new_row = {"columns": feature, "scores": result.pvalues[-1]}

                # print(f"feature p_val:  {result.pvalues[-1]}")

                new_row_df = pd.DataFrame([new_row])

                scores_dict = pd.concat([scores_dict, new_row_df], ignore_index=True)

        # pd.set_option('display.float_format', '{:.2f}'.format)
        # DO NAPRAWY
        id_min = scores_dict['scores'].idmin(axis=0)

        # print(id_min)
        return scores_dict['columns'][id_min], scores_dict['scores'][id_min]

    def model_criterion_eval(self,
                             y,
                             X_train,
                             best_model_score,
                             selected_features,
                             remaining_features,
                             not_dropped_features,
                             feature,
                             ):

        test_feature_set = list(filter(lambda idx: idx != feature, selected_features))

        # print(f"mc features set: {test_feature_set}")

        model_criterion_val = super().count_model_criterion(X_train,
                                                            y,
                                                            test_feature_set)
        # print(f"model criterion: {model_criterion_val}")

        if self.model_criterion == "AIC":

            if model_criterion_val > best_model_score:
                best_model_score = model_criterion_val

                selected_features = test_feature_set
                remaining_features = list(filter(lambda idx: idx != feature, remaining_features))

                super().append_log(self.model_criterion,
                                   model_criterion_val,
                                   selected_features)
            else:
                print('Cannot select better subset using AIC')
                remaining_features = list(filter(lambda idx: idx != feature, remaining_features))
                not_dropped_features.append(feature)

        elif self.model_criterion == "BIC":

            if model_criterion_val > best_model_score:
                best_model_score = model_criterion_val

                selected_features = test_feature_set
                remaining_features = list(filter(lambda idx: idx != feature, remaining_features))

                super().append_log(self.model_criterion,
                                   model_criterion_val,
                                   selected_features)

            else:
                print('Cannot select better subset using BIC')
                remaining_features = list(filter(lambda idx: idx != feature, remaining_features))
                not_dropped_features.append(feature)

        elif self.model_criterion == "pseudo_R_square":

            print(f"test_feature_set: {test_feature_set}")
            print(f"remaining_features: {remaining_features}")
            print(f"not_dropped_features: {not_dropped_features}")

            if model_criterion_val < best_model_score:
                best_model_score = model_criterion_val

                selected_features = test_feature_set
                remaining_features = list(filter(lambda idx: idx != feature, remaining_features))

                super().append_log(self.model_criterion,
                                   model_criterion_val,
                                   selected_features)
            else:
                print('Cannot select better subset using pseudo R-square')
                remaining_features = list(filter(lambda idx: idx != feature, remaining_features))
                not_dropped_features.append(feature)

        else:
            raise ValueError("Invalid stopping criterion. Correct values: 'AIC', 'BIC' or pseudo_R_square.")

        # self.logs_df.to_csv("log_df.csv", decimal=".", sep="|", mode='a', index=False, header=False)

        return best_model_score, selected_features, remaining_features, not_dropped_features

    def evaluate_model(self,
                       df: pd.DataFrame(),
                       df_pre_split: pd.DataFrame() = None,
                       pre_split: bool = None):

        X_train, X_test, y_train, y_test, best_subset = self.select_subset(df=df,
                                                                           df_pre_split=df_pre_split,
                                                                           pre_split=pre_split)
        """
        best_subset = self.logs_df["Selected_features"].loc[
            self.logs_df["Model_criterion_value"].idxmax() if self.logs_df["Model_criterion"].iloc[
                                                                  0] == "pseudo_R_square"
            else self.logs_df["Model_criterion_value"].idxmin()
        ]
        """
        print(f"best_subset: {best_subset}")

        X_train = super().select_based_on_idx(X_train, best_subset)
        X_test = super().select_based_on_idx(X_test, best_subset)

        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)

        metrics_calculator = SelectedMetrics()
        return metrics_calculator.return_conf_matrix_related_metrics(y_true=y_test, y_predict=y_predict)


# class HybridStepwiseSelection:

class Lasso:
    @staticmethod
    def perform_lasso_logistic_regression(df: pd.DataFrame(),
                                          df_pre_split: pd.DataFrame() = None,
                                          pre_split=False):

        metrics_calculator = SelectedMetrics()

        if pre_split:
            X_train, X_test, y_train, y_test = ds.split_data(data_set=df,
                                                             data_set_if_pre=df_pre_split,
                                                             pre_split=pre_split)
            model = LogisticRegression(penalty='l1', solver='liblinear')
            train = model.fit(X_train, y_train)

            y_predict = train.predict(X_test)
            return metrics_calculator.return_conf_matrix_related_metrics(y_true=y_test, y_predict=y_predict)

        if not pre_split:
            X_train, X_test, y_train, y_test = ds.split_data(data_set=df, pre_split=pre_split)
            model = LogisticRegression(penalty='l1', solver='liblinear')

            train = model.fit(X_train, y_train)

            y_predict = train.predict(X_test)

            return metrics_calculator.return_conf_matrix_related_metrics(y_true=y_test, y_predict=y_predict)


class CrossValidation:
    """
    DataSet 4 is not supprted (pre_split from vendor)
    """

    @staticmethod
    def eval_cross_validation_train(df: pd.DataFrame()):
        metrics_calculator = SelectedMetrics()
        split_count = 10

        X = df.drop(['y'], axis=1)
        y = df['y']

        model = LogisticRegression()

        k_folds = KFold(n_splits=split_count)

        tp = tn = fp = fn = []

        print(k_folds.split(X))

        for train_index, test_index in k_folds.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate confusion matrix for the current fold
            cm = confusion_matrix(y_test, y_pred)

            # Update true positives, true negatives, false positives, and false negatives
            tp.append(cm[1, 1])
            tn.append(cm[0, 0])
            fp.append(cm[0, 1])
            fn.append(cm[1, 0])

        return metrics_calculator.return_conf_matrix_related_metrics_for_cross(_tp=tp, _tn=tn, _fp=fp, _fn=fn)


class FeaturePermutation:
    @staticmethod
    def perform_selection_based_on_permutation(df: pd.DataFrame(),
                                               df_pre_split: pd.DataFrame() = None,
                                               pre_split=False):

        model = LogisticRegression()

        if pre_split:
            X_train, X_test, y_train, y_test = ds.split_data(data_set=df,
                                                             data_set_if_pre=df_pre_split,
                                                             pre_split=pre_split)

            perm_imp = permutation_importance(model,
                                              X_test,
                                              y_test,
                                              n_repeats=30,
                                              random_state=42)
            return

        if not pre_split:
            X_train, X_test, y_train, y_test, columns = ds.split_data(data_set=df,
                                                                      data_set_if_pre=df_pre_split,
                                                                      pre_split=pre_split,
                                                                      dict_columns=True)

            # df_columns = ds.return_columns(data_set=df, pre_split=False)

            train = model.fit(X_train, y_train)
            y_predict = train.predict(X_test)

            conf_matrix_metrics = SelectedMetrics().return_conf_matrix_related_metrics(y_test=y_test,
                                                                                       y_predict=y_predict)
            print(conf_matrix_metrics)
            orginal_accuracy = conf_matrix_metrics['accuracy']
            orginal_f1 = conf_matrix_metrics['f1_score']

            X_test_with_columns = pd.DataFrame(data=X_test, columns=columns)

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
    feature criterion should be "p_value"
    feed model only with variables which match feature criterion
    Data set 4 cannot run due to lack of memory
    """

    def __init__(self,
                 feature_criterion: str = "p_value",
                 criterion_val=float):
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

        selected_features_idx = [idx for idx, feature in enumerate(result.pvalues) if
                                 feature <= float(self.criterion_val)]

        print(selected_features_idx)

        X_train = X_train[:, selected_features_idx]
        X_test = X_test[:, selected_features_idx]

        return X_train, X_test, y_train, y_test

    def evaluate_model(self,
                       df: pd.DataFrame(),
                       df_pre_split: pd.DataFrame() = None,
                       pre_split: bool = None):
        X_train, X_test, y_train, y_test = self.select_subset(df=df,
                                                              df_pre_split=df_pre_split,
                                                              pre_split=pre_split)

        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)

        metrics_calculator = SelectedMetrics()
        return metrics_calculator.return_conf_matrix_related_metrics(y_true=y_test, y_predict=y_predict)


class SelectedMetrics:

    def __init__(self):
        self.metrics = pd.DataFrame({'recall': [float()],
                                     'precision': [float()],
                                     'specificity': [float()],
                                     'negative_predictive_value': [float()],
                                     'accuracy': [float()],
                                     'balanced_accuracy': [float()],
                                     'f1_score': [float()]})

    def return_conf_matrix_related_metrics(self,
                                           y_true: list,
                                           y_predict: list):
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_predict).ravel()

        self.calculate_metrics(tn=tn,
                               fp=fp,
                               fn=fn,
                               tp=tp)
        # print(f"cm: {confusion_matrix(y_true=y_true, y_pred=y_predict)}")

        return self.metrics

    def return_conf_matrix_related_metrics_for_cross(self,
                                                     _tn: list,
                                                     _fp: list,
                                                     _fn: list,
                                                     _tp: list):

        for tn, fp, fn, tp in zip(_tn, _fp, _fn, _tp):
            self.calculate_metrics(tn=tn,
                                   fp=fp,
                                   fn=fn,
                                   tp=tp)

        mean_values = self.metrics.iloc[:-1].mean()
        self.metrics = pd.DataFrame(mean_values).T

        return self.metrics

    def calculate_metrics(self,
                          tn: int,
                          fp: int,
                          fn: int,
                          tp: int):
        # recall known as sensitivity as well
        metrics_formula = {'recall': tp / (tp + fn),
                           'precision': tp / (tp + fp),
                           'specificity': tn / (tn + fp),
                           'negative_predictive_value': tn / (tn + fn),
                           'accuracy': (tp + tn) / (tp + tn + fp + tn),
                           'balanced_accuracy': ((tp / (tp + fn)) + (tn / (tn + fp))) / 2,
                           'f1_score': 2 * (
                                   ((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn))))
                           }

        new_row_df = pd.DataFrame([metrics_formula])

        self.metrics = pd.concat([self.metrics, new_row_df], ignore_index=True)

        if (self.metrics.loc[0] == 0).all():
            self.metrics.drop(index=0, axis=0, inplace=True)

        self.metrics.reset_index(drop=True, inplace=True)

        pass
