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
cross_val = ml.CrossValidation()
fp = ml.FeaturePermutation()
bf = ml.BruteForce(criterion_val=0.6)
fss = ml.ForwardStepwiseSelection(model_criterion='AIC', feature_criterion='p_value')
# bss = ml.BackwardStepwiseSelection(model_criterion='AIC', feature_criterion='p-value')

# d4_train_standard = d4.preprocess_train_data(scaler_type='StandardScaler')
# d4_test_standard = d4.preprocess_test_data(scaler_type='StandardScaler')
# xd = lasso.perform_lasso_logistic_regression(df=d4_train_standard, df_pre_split=d4_test_standard, pre_split=True)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = time.time()
    print("Pipeline start")

    d1_standard = d1.preprocess_data(scaler_type='StandardScaler')
    # d2_standard = d2.preprocess_data(scaler_type='StandardScaler')
    # d3_standard = d3.preprocess_data(scaler_type='StandardScaler')

    # d4_train_standard = d4.preprocess_train_data(scaler_type='StandardScaler')
    # d4_test_standard = d4.preprocess_test_data(scaler_type='StandardScaler')

    clc = ml.SelectedMetrics()

    xd = lasso.perform_lasso_logistic_regression(d1_standard)
    clc.append_metrics("data_set1", "lasso", xd)
    xd2 = bf.evaluate_model(d1_standard)
    dx = clc.append_metrics("data_set_1", "brute_force", xd2)
    print(dx)
    """
    d4_train_standard_comb = comb_generator.generate_combinations(d4_train_standard)
    print(len(d4_train_standard_comb))
    for combination in d4_train_standard_comb:
        train = comb_generator.return_sub_df(d4_train_standard, combination)
        test = comb_generator.return_sub_df(d4_train_standard, combination)
        xd = fss.evaluate_model(df=train, df_pre_split=test, pre_split=True)
    # xd = bf.evaluate_model(sample_dataset)
    """
    print(xd)

    end = time.time()
    print(f"Pipeline exec time: {(end - start) / 60} min")

    # print(xd)




