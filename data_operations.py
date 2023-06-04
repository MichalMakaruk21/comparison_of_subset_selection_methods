from IPython.display import display
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

# Scaler Selector Class
class ScalerSelector:
    def __init__(self, scaler_type):
        self.scaler_type = scaler_type

    def get_scaler(self):
        if self.scaler_type == 'StandardScaler':
            return StandardScaler()
        elif self.scaler_type == 'MinMaxScaler':
            return MinMaxScaler()
        elif self.scaler_type == 'RobustScaler':
            return RobustScaler()
        elif self.scaler_type == 'Normalizer':
            return Normalizer()
        else:
            raise ValueError('Invalid scaler type specified.')


# Class for Dataset 1
class Dataset1:
    def __init__(self):
        self.file_path = 'dataset1.csv'

    def read_data(self):
        return pd.read_csv(self.file_path)

    def preprocess_data(self, scaler_type: str):
        data = self.read_data()
        X = data.drop('target', axis=1)
        y = data['target']

        with ScalerSelector(scaler_type).get_scaler() as scaler:
            X_scaled = scaler.fit_transform(X)

        return X_scaled, y


# Class for Dataset 2
class Dataset2:
    def __init__(self):
        self.file_path = 'dataset2.csv'

    def read_data(self):
        return pd.read_csv(self.file_path)

    def preprocess_data(self, scaler_type: str):
        data = self.read_data()
        X = data.drop('target', axis=1)
        y = data['target']

        with ScalerSelector(scaler_type).get_scaler() as scaler:
            X_scaled = scaler.fit_transform(X)

        return X_scaled, y


# Class for Dataset 3
class Dataset3:
    def __init__(self):
        self.file_path = 'dataset3.csv'

    def read_data(self):
        return pd.read_csv(self.file_path)

    def preprocess_data(self, scaler_type: str):
        data = self.read_data()
        X = data.drop('target', axis=1)
        y = data['target']

        with ScalerSelector(scaler_type).get_scaler() as scaler:
            X_scaled = scaler.fit_transform(X)

        return X_scaled, y


# Class for Dataset 4
class Dataset4:
    def __init__(self):
        self.file_path = 'dataset4.csv'

    def read_data(self):
        return pd.read_csv(self.file_path)

    def preprocess_data(self, scaler_type: str):
        data = self.read_data()
        X = data.drop('target', axis=1)
        y = data['target']

        with ScalerSelector(scaler_type).get_scaler() as scaler:
            X_scaled = scaler.fit_transform(X)

        return X_scaled, y
