import pandas as pd
import numpy as np
import re
from IPython.display import display

import data_operations as d
import ML_operations as ml
import matplotlib as plt
d1 = d.DataSet1()
d2 = d.DataSet2()
d3 = d.DataSet3()
d4 = d.DataSet4()

lasso = ml.Lasso()

fss = ml.ForwardStepwiseSelection(model_criterion='AIC', feature_criterion='p-value')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    df = d1.preprocess_data(scaler_type='StandardScaler')
    # lasso_metrics = lasso.perform_lasso_logistic_regression(df=df, pre_split=False)
    # sub_df = fss.select_subset(df)
    display(lasso.perform_lasso_logistic_regression(df=df, pre_split=False))
    # print(sub_df.columns)
    # print(sub_df.head())

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