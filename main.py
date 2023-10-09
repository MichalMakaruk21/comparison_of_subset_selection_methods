import pandas as pd
import numpy as np
import re
from IPython.display import display
import itertools

import data_operations as d
import ML_operations as ml
import matplotlib as plt
import time

# Instances of classes containing data
d1 = d.DataSet1()
d2 = d.DataSet2()
d3 = d.DataSet3()
d4 = d.DataSet4()

comb_generator = d.SubDataFrameGenerator()

lasso = ml.Lasso()
kross_val = ml.CrossValidation()

feature_importance = ml.FeaturePermutation()
bss = ml.BackwardStepwiseSelection(model_criterion='AIC', feature_criterion='p-value')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    sample_dataset = d1.preprocess_data(scaler_type='StandardScaler')
    # sample_dataset = d2.preprocess_data(scaler_type='StandardScaler')
    # sample_dataset = d3.preprocess_data(scaler_type='StandardScaler')
    # sd_A = d4.preprocess_train_data(scaler_type='StandardScaler')
    # sd_B = d4.preprocess_test_data(scaler_type='StandardScaler')

    # fss.select_subset(data_set=sample_dataset)

    # features_combination = comb_generator.generate_combinations(df=sample_dataset)
    # display(features_combination_series)
    """
    for comb in features_combination:
        sub_df = comb_generator.return_sub_df(df=sample_dataset, combination=comb)
        df_logs = fss.select_subset(data_set=sub_df)
        # df_logs = bss.select_subset(data_set=sub_df)
        display(df_logs) 
        """


    # fss = ml.ForwardStepwiseSelection(model_criterion='AIC', feature_criterion='pseudo-R-square')

    start = time.time()
    print("Pipeline start")
    # df_logs = fss.select_subset(data_set=sample_dataset)
    # log = fss.logs_df

    # bf = ml.BruteForce(feature_criterion='p_value', criterion_val='0.2')
    # ll = bf.select_subset(sample_dataset)

    #  bf = ml.BruteForce(feature_criterion='p_value', criterion_val='0.2')
    # ll = bf.select_subset(df=sd_A, df_pre_split=sd_B, pre_split=True)

    # fss = ml.ForwardStepwiseSelection(model_criterion='AIC', feature_criterion='p_value')
    # forward_subset_selection_ds1 = fss.evaluate_model(sample_dataset)

    bss = ml.BackwardStepwiseSelection(model_criterion='AIC', feature_criterion='p_value')
    xd = bss.select_subset(sample_dataset)

    print(xd)
    # log_df.to_csv("log_df.csv", decimal=".", sep="|")

    # bf = ml.BruteForce(feature_criterion='p_value', criterion_val='0.2')
    # bf = ml.BruteForce(feature_criterion='pseudo_R_square', criterion_val='0.2')
    # bf.select_subset(df=sample_dataset)

    """
    for ds in ["d1", "d2", "d3"]:
        x = eval(f"{ds}.preprocess_data(scaler_type='StandardScaler')")
        ll = comb_generator.generate_combinations(x)
        print(len(ll))
"""
    end = time.time()
    print(f"Pipeline exec time: {end - start}")
    # print(ll)
    # df_logs.to_csv("fss_log.csv", sep="|")




    """
    print(df)

    for x in df.columns:

        # print(df[x].describe())
        #print(x)
        # print(df[x].dtype)

        if df[x].isnull().values.any():
            print(f"---------The column {x} contains null or NaN values.")
            print(df[x].dtype)
        else:
            # print(f"The column {x} does NOT contain null or NaN values.")
            print('---')

        num_unique_values = df[x].nunique()
        print(f"The column has {num_unique_values} unique value(s).")

    #for id ,val in zip(df.index, df['Attr63']):
    #   print(f'id: {id} val: {val}')
"""

"""
# -------------------------------
        for x in X.columns:

            # print(df[x].describe())
            # print(x)
            # print(df[x].dtype)

            if X[x].isnull().values.any():
                print(f"---------The column {x} contains null or NaN values.")
            else:
                print(f"The column {x} does NOT contain null or NaN values.")
            print(X[x].dtype)
            num_unique_values = X[x].nunique()
            print(f"The column has {num_unique_values} unique value(s).")

            # for id, val in zip(X.index, X['Attr24']):
            #     print(f'id: {id} val: {val}')

# --------------------------------------"""

# lasso_metrics = lasso.perform_lasso_logistic_regression(df=df, pre_split=False)
# sub_df = fss.select_subset(df)
# display(lasso.perform_lasso_logistic_regression(df=df, pre_split=False))
# display(kross_val.perform_kross_validation_train(df))
# display(feature_importance.perform_selection_based_on_permutation(df, pre_split=False))
# print(sub_df.columns)
# print(sub_df.head())
