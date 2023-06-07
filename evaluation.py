import pandas as pd
import data_operations as do
import ML_operations as ml

nomalisation_list = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer']
feature_rate = ['p_value', 'pseudo_R_square']
criterion_list = ['AIC', 'BIC']

permutation_criterion_list = ['acc', 'F1', 'pseudo_R_square']

ds1 = do.DataSet1()
ds2 = do.DataSet2()
ds3 = do.DataSet3()
ds4 = do.DataSet4()

"""
subset selection methods
"""

data_set_list = [ds1, ds2, ds3, ds4]

for data_set in data_set_list:
    for norm in nomalisation_list:

        if data_set != ds4:
            test_data = data_set.preprocess_train_data(scaler_type=norm)

            ds = do.DataSplitter().split_data(data_set=data_set, pre_split=False)
        else:
            data = data_set.preprocess_data(scaler_type=norm)
            ds = do.DataSplitter().split_data(data_set=data_set, pre_split=False)
# ml.