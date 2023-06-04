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

    """
    Implementuje dane z badania:  "Artificial intelligence in breast cancer screening:
                                   Primary care provider preferences" (2020-07-16)
    Link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EX4NG2
    Badanie opini kobiet (dane ankietowe) na temat wykorzystania technogi AI podczas badań lekarskich.
    """
    def __init__(self):
        self.file_path = r'data/Artificial_intelligence_in_breast_cancer_screening_Primary_care_provider_preferences/ai_pcp_processed-1.csv'

    def read_data(self):
        return pd.read_csv(self.file_path, sep=',', header=0)

    def preprocess_data(self, scaler_type: str):
        data = self.read_data()
        """
        zmienna niezależna: "tech_negative"
        0 - opinia nagatywna
        1 - opinia pozytywna
        """
        # usunięcie kolumny identyfikującej uczestnika ankiety
        data = data.drop('id', axis=1)
        X = data.drop('target', axis=1)
        y = data['target']

        # dane nie wymagają standaryzacji ponieważ zawierają same wartości kategoryczne
        # np. -1, 0, 1 (zmiana stosunku do rozwiązań AI), 15 (staż pracy lekarza prowadzącego badnie)

        return X, y


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
