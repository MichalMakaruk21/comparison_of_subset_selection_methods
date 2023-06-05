import pandas as pd

from data_operations import Dataset1
from data_operations import Dataset4
import matplotlib as plt
# d = Dataset1()
# X, y = d.preprocess_data(scaler_type='StandardScaler')

d = Dataset4()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    numerical_cols = ['age', 'industry_code', 'occupation_code', 'wage_per_hour', 'capital_gains', 'capital_losses',
                      'divdends_from_stocks', 'num_persons_worked_for_employer', 'own_business_or_self_employed',
                      'veterans_benefits', 'weeks_worked_in_year', 'year']

    df = d.read_train_data()
    df_check = df[numerical_cols]

    print(df_check.dtypes)
