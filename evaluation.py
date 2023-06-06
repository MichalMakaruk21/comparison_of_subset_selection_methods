import pandas as pd
import data_operations as do
import ML_operations as ML

nomalisation_list = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer']
feature_rate = ['p_value', 'pseudo_R_square']
criterion_list = ['AIC', 'BIC']

permutation_criterion_list = ['acc', 'F1', 'pseudo_R_square']

ds1 = do.DataSet1()
ds2 = do.DataSet2()
ds3 = do.DataSet3()
ds4 = do.DataSet4()

data_set_list = [ds1, ds2, ds3, ds4]

for data_set in data_set_list:
    for norm in nomalisation_list:

        data = data_set.preprocess_data(scaler_type=norm)


