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
# fss = ml.ForwardStepwiseSelection(model_criterion='AIC', feature_criterion='p_value')
# bss = ml.BackwardStepwiseSelection(model_criterion='AIC', feature_criterion='p-value')




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = time.time()
    print("Pipeline start")

    sample_dataset = d1.preprocess_data(scaler_type='StandardScaler')

    # xd = cross_val.eval_cross_validation_train(sample_dataset)

    xd = fp.evaluate_model(sample_dataset)
    print(xd)

    end = time.time()
    print(f"Pipeline exec time: {end - start}")

    # print(xd)




