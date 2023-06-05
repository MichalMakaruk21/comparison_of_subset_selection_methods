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
        data = data.drop('id', axis=1)
        X = data.drop('tech_negative', axis=1)
        y = data['tech_negative']

        # dane nie wymagają standaryzacji ponieważ zawierają same wartości kategoryczne
        # np. -1, 0, 1 (zmiana stosunku do rozwiązań AI), 15 (staż pracy lekarza prowadzącego badnie)

        return X, y


# Class for Dataset 2
class Dataset2:
    """
    Implementuje dane zbierane przez Sebastiana Tomczaka na przestrzeni 12 lat
    reator: Sebastian Tomczak
    Department of Operations Research, Wroclaw University of Science and Technology, wybrzeże Wyspiaańskiego 27, 50-370, Wroclaw, Poland

    link: https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data

    Wybrałem naliczeniejszy zbór danych, charakteyzujący się bankrupstwem po 3 latach działalności firmy
    """
    def __init__(self):
        self.file_path = 'data/Polish_companies_bankruptcy/polish_companies_bankruptcy_3.csv'

    def read_data(self):
        return pd.read_csv(self.file_path, sep=',', index_col='id', low_memory=False)

    def preprocess_data(self, scaler_type: str):
        data = self.read_data()
        X = data.drop('class', axis=1)
        y = data['class']

        # usuń ? i obsłuż puste dane i 0
        # X = X.replace(r'^\s*$', np.nan, regex=True)

        with ScalerSelector(scaler_type).get_scaler() as scaler:
            X_scaled = scaler.fit_transform(X)

        return X_scaled, y


# Class for Dataset 3
class Dataset3:
    """
    Implementuje dane z badania: Transfer Learning with Partial Observability Applied to Cervical Cancer Screening.'
                                 Iberian Conference on Pattern Recognition and Image Analysis.
                                 Springer International Publishing, 2017.
                                 Kelwin Fernandes, Jaime S. Cardoso, and Jessica Fernandes.

    link: https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29

    Badania uwzględniły 4 zmienne niezależne, będące sposobami wykrywania raka szyjki macicy:
        -Hinselmann
        -Schiller
        -Cytology
        -Biopsy (wybrana jako zmienna niezależna dla moich badań)
    """
    def __init__(self):
        self.file_path = 'data/Cervical_cancer_(Risk Factors)/risk_factors_cervical_cancer.csv'

    def read_data(self):
        return pd.read_csv(self.file_path, sep=',')

    def preprocess_data(self, scaler_type: str):
        data = self.read_data()
        X = data.drop('Biopsy', axis=1)
        y = data['Biopsy']

        with ScalerSelector(scaler_type).get_scaler() as scaler:
            X_scaled = scaler.fit_transform(X)

        return X_scaled, y


# Class for Dataset 4
class Dataset4:
    """
    Implementuje dane z udostępnione przez U.S. Census Bureau
    Temat: Census-Income (KDD) Data Set
    link: https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)
    Dane zostały zebrane poprzez badanie ankietowe i dają szerokie spojrzenie na zależność czynników
    społeczno-ekonomicznych na zarobki ankietowanych. Zagregowane przez dostawcę dane mają odpowiadać na pytanie
    "Czy dana osoba będzie zarabiać powyżej, czy poniżej 50 000 tys dolarów rocznie?".

    Pomimo że dane są ze źródła open-source, były cytowane i wykożystywane w artykułach naukowych.
    m.in. Masahiro Terabe and Takashi Washio and Hiroshi Motoda.
          The Effect of Subsampling Rate on S 3 Bagging Performance. Mitsubishi Research Institute.
    """
    def __init__(self):
        self.file_path_train_data = 'data/Census_Income_KDD_Prediction/census_income_data.csv'
        self.file_path_test_data = 'data/Census_Income_KDD_Prediction/census_income_test.csv'
        self.column_names = ['age', 'class_of_worker', 'industry_code', 'occupation_code', 'education', 'wage_per_hour',
                             'enrolled_in_edu_inst_last_wk', 'marital_status', 'major_industry_code',
                             'major_occupation_code', 'race', 'hispanic_Origin', 'sex', 'member_of_a_labor_union',
                             'reason_for_unemployment', 'full_or_part_time_employment_stat', 'capital_gains',
                             'capital_losses', 'divdends_from_stocks', 'tax_filer_status',
                             'region_of_previous_residence', 'state_of_previous_residence',
                             'detailed_household_and_family_stat',
                             'detailed_household_summary_in_household', 'instance_weight',
                             'migration_code_change_in_msa', 'migration_code_change_in_reg',
                             'migration_code_move_within_reg', 'live_in_this_house_1_year_ago',
                             'migration_prev_res_in_sunbelt', 'num_persons_worked_for_employer',
                             'family_members_under_18', 'country_of_birth_father', 'country_of_birth_mother',
                             'country_of_birth_self', 'citizenship', 'own_business_or_self_employed',
                             'fill_inc_questionnaire_for_veterans_admin', 'veterans_benefits', 'weeks_worked_in_year',
                             'year', 'income']
    def read_train_data(self):
        df = pd.read_csv(self.file_path_train_data, sep=',', names=self.column_names, header=0)
        df['income'] = df['income'].replace({' - 50000.': 0, ' 50000+.': 1})
        return df
    def read_test_data(self):
        df = pd.read_csv(self.file_path_test_data, sep=',', names=self.column_names, header=0)
        df['income'] = df['income'].replace({' - 50000.': 0, ' 50000+.': 1})
        return df

    def preprocess_test_data(self, scaler_type: str):
        train_data = self.read_train_data()

        X = train_data.drop('income', axis=1)
        y = train_data['income']

        # dataset contains empty values saved as "?" only in categorical columns
        # these column will be transformed into dummy variables or dropped
        # zapytaj !!!!!!!!!!!

        # X = X.replace('?', str("NaN"))
        X = X.replace('?', np.nan)

        X_dummy = pd.get_dummies(X, dummy_na=False, drop_first=True)

        with ScalerSelector(scaler_type).get_scaler() as scaler:
            X_scaled = scaler.fit_transform(X_dummy)

        return X_scaled, y

    def preprocess_train_data(self, scaler_type: str):
        test_data = self.read_train_data()

        X = test_data.drop('income', axis=1)
        y = test_data['income']

        X = X.replace('?', np.nan)
        X_dummy = pd.get_dummies(X, dummy_na=False, drop_first=True)

        with ScalerSelector(scaler_type).get_scaler() as scaler:
            X_scaled = scaler.fit_transform(X_dummy)

        return X_scaled, y
