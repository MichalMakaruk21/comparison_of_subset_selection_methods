import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.model_selection import train_test_split
import itertools


class DataSet(object):
    def read_data(self) -> pd.DataFrame():
        pass

    def preprocess_data(self, scaler_type: str) -> pd.DataFrame():
        pass

    def read_train_data(self) -> pd.DataFrame():
        pass

    def read_test_data(self) -> pd.DataFrame():
        pass

    def preprocess_train_data(self, scaler_type: str) -> pd.DataFrame():
        pass

    def preprocess_test_data(self, scaler_type: str) -> pd.DataFrame():
        pass


class DataSet1(DataSet):
    """
    Implementuje dane z badania:  "Artificial intelligence in breast cancer screening:
                                   Primary care provider preferences" (2020-07-16)
    Link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EX4NG2
    Badanie opini kobiet (dane ankietowe) na temat wykorzystania technogi AI podczas badań lekarskich.

    brak pystych danych
    dane charkateryzują się transgforamcją danych katerorycznych na indeksy i
    np. -1, 0, 1 (zmiana stosunku do rozwiązań AI), 15 (staż pracy lekarza prowadzącego badnie)
    """

    def __init__(self):
        self.file_path = r'data/Artificial_intelligence_in_breast_cancer_screening_Primary_care_provider_preferences/ai_pcp_processed-1.csv'

    def read_data(self) -> pd.DataFrame():
        return pd.read_csv(self.file_path, sep=',', header=0)

    def preprocess_data(self, scaler_type: str) -> pd.DataFrame():
        data = self.read_data()
        """
        zmienna niezależna: "tech_negative"
        0 - opinia nagatywna
        1 - opinia pozytywna
        """
        # drop responder ID
        data = data.drop('id', axis=1)

        X = data.drop('tech_negative', axis=1)
        y = data['tech_negative']

        X = X.interpolate(method='linear')

        scaler = ScalerSelector().get_scaler(scaler_type=scaler_type)
        X_scaled = scaler.fit_transform(X)

        df = ScaledDataFrameBuilder().get_df_from_preprocess_data(x_scaled=X_scaled, y=y, columns=X.columns)

        return df


class DataSet2(DataSet):
    """
    Implementuje dane zbierane przez Sebastiana Tomczaka na przestrzeni 12 lat
    reator: Sebastian Tomczak
    Department of Operations Research, Wroclaw University of Science and Technology, wybrzeże Wyspiaańskiego 27, 50-370, Wroclaw, Poland

    link: https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data

    Wybrałem naliczeniejszy zbór danych, charakteyzujący się bankrupstwem po 3 latach działalności firmy
    """

    def __init__(self):
        self.file_path = 'data/Polish_companies_bankruptcy/polish_companies_bankruptcy_3.csv'

    def read_data(self) -> pd.DataFrame():
        return pd.read_csv(self.file_path, sep=',', index_col='id', low_memory=False, encoding='utf-8')

    def preprocess_data(self, scaler_type: str) -> pd.DataFrame():
        data = self.read_data()
        X = data.drop('class', axis=1)
        y = data['class']

        # nan values are represented by "?"
        # replace '?" with 'nan' and fill using linear interpolation
        X = X.replace('?', np.nan)
        X = X[X.columns.tolist()].astype('float64')
        X = X.interpolate(method='linear', axis=1)

        scaler = ScalerSelector().get_scaler(scaler_type=scaler_type)
        X_scaled = scaler.fit_transform(X)

        df = ScaledDataFrameBuilder().get_df_from_preprocess_data(x_scaled=X_scaled, y=y, columns=X.columns)

        return df


class DataSet3(DataSet):
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
        self.file_path = 'data/Cervical_cancer_Risk_Factors/risk_factors_cervical_cancer.csv'

    def read_data(self) -> pd.DataFrame():
        return pd.read_csv(self.file_path, sep=',')

    def preprocess_data(self, scaler_type: str) -> pd.DataFrame():
        data = self.read_data()

        # Usunięcie z powodu braku wartości w ponad 93%
        data = data.drop(['STDs: Time since first diagnosis',
                          'STDs: Time since last diagnosis'], axis=1)

        # większość pustych wartości jest skumulowanych wzdłóż rekordu,
        # (rekord który ma chociaż jedną pustą wartość, w zdecydowanej większości
        # ma również braki w reszcie rekorów dotyczących STD)
        data = data.replace('?', np.nan).dropna(axis=0)

        # usunięcie innych potencjalnych zmiennych objaśnianych
        data = data.drop(['Hinselmann',
                          'Schiller',
                          'Citology'], axis=1)

        X = data.drop('Biopsy', axis=1)
        y = data['Biopsy']

        scaler = ScalerSelector().get_scaler(scaler_type=scaler_type)
        X_scaled = scaler.fit_transform(X)

        df = ScaledDataFrameBuilder().get_df_from_preprocess_data(x_scaled=X_scaled, y=y, columns=X.columns)

        return df


class DataSet4(DataSet):
    """
    Implementuje dane z udostępnione przez U.S. Census Bureau
    Temat: Census-Income (KDD) Data Set
    link:
    Dane zostały zebrane https://archive.ics.uci.edu/dataset/117/census+income+kdd
    poprzez badanie ankietowe i dają szerokie spojrzenie na zależność czynników
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

    def read_train_data(self) -> pd.DataFrame():
        df = pd.read_csv(self.file_path_train_data, sep=',', names=self.column_names, header=0)
        df['income'] = df['income'].replace({' - 50000.': 0, ' 50000+.': 1})
        return df

    def read_test_data(self) -> pd.DataFrame():
        df = pd.read_csv(self.file_path_test_data, sep=',', names=self.column_names, header=0)
        df['income'] = df['income'].replace({' - 50000.': 0, ' 50000+.': 1})
        return df

    def preprocess_train_data(self, scaler_type: str) -> pd.DataFrame():
        train_data = self.read_train_data()

        X = train_data.drop('income', axis=1)
        y = train_data['income']

        # dataset contains empty values saved as "?" only in categorical columns
        # these column will be transformed into dummy variables or dropped
        # zapytaj !!!!!!!!!!!

        X = X.replace(['Not in universe', '?', 'Not in universe or children', 'Do not know',
                       'Not in universe under 1 year old', 'NA'], np.nan)

        X['hispanic_Origin'].fillna(X['hispanic_Origin'].mode()[0], inplace=True)
        X["cap_gain"] = np.where(X['capital_gains'] > 0, 1, 0)
        X["cap_loss"] = np.where(X['capital_losses'] > 0, 1, 0)

        X_dummy = pd.get_dummies(X, dummy_na=False, drop_first=True)
        scaler = ScalerSelector().get_scaler(scaler_type=scaler_type)
        X_scaled = scaler.fit_transform(X_dummy)

        df = ScaledDataFrameBuilder().get_df_from_preprocess_data(x_scaled=X_scaled, y=y, columns=X_dummy.columns)
        print('preprocess_finished')
        return df

    def preprocess_test_data(self, scaler_type: str) -> pd.DataFrame():
        test_data = self.read_train_data()

        X = test_data.drop('income', axis=1)
        y = test_data['income']

        X = X.replace(['Not in universe', '?', 'Not in universe or children', 'Do not know',
                       'Not in universe under 1 year old', 'NA'], np.nan)

        X['hispanic_Origin'].fillna(X['hispanic_Origin'].mode()[0], inplace=True)
        X["cap_gain"] = np.where(X['capital_gains'] > 0, 1, 0)
        X["cap_loss"] = np.where(X['capital_losses'] > 0, 1, 0)

        X_dummy = pd.get_dummies(X, dummy_na=False, drop_first=True)

        scaler = ScalerSelector().get_scaler(scaler_type=scaler_type)
        X_scaled = scaler.fit_transform(X_dummy)

        df = ScaledDataFrameBuilder().get_df_from_preprocess_data(x_scaled=X_scaled, y=y, columns=X_dummy.columns)

        return df


class ScaledDataFrameBuilder:
    def get_df_from_preprocess_data(self,
                                    x_scaled: pd.DataFrame(),
                                    y: pd.DataFrame(),
                                    columns: list) -> pd.DataFrame():
        df = pd.DataFrame(data=x_scaled, columns=[x.strip() for x in columns])
        df['y'] = y.tolist()
        return df


class ScalerSelector:

    @staticmethod
    def get_scaler(scaler_type: str):

        if scaler_type == 'StandardScaler':
            return StandardScaler()
        elif scaler_type == 'MinMaxScaler':
            return MinMaxScaler()
        elif scaler_type == 'RobustScaler':
            return RobustScaler()
        elif scaler_type == 'Normalizer':
            return Normalizer(norm='l1')
        else:
            raise ValueError('Invalid scaler type specified.')


class DataSplitter:
    """
    Objective split parameter setter
    """

    @staticmethod
    def split_data(data_set: pd.DataFrame() = None,
                   data_set_if_pre: pd.DataFrame() = None,
                   test_size=0.2,
                   random_state=21,
                   pre_split: bool = False,
                   dict_columns: bool = False):
        if not pre_split:
            X = np.array((data_set.drop(['y'], axis=1)))
            y = np.array((data_set['y']))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            if dict_columns is True:
                return X_train, X_test, y_train, y_test, list(data_set.columns[:-1])
            else:
                return X_train, X_test, y_train, y_test

        elif pre_split:
            X_train = np.array(data_set.drop(['y'], axis=1))
            X_test = np.array(data_set_if_pre.drop(['y'], axis=1))
            y_train = np.array(data_set['y'])
            y_test = np.array(data_set_if_pre['y'])

            if dict_columns is True:
                return X_train, X_test, y_train, y_test, list(data_set.columns[:-1])
            else:
                return X_train, X_test, y_train, y_test
        else:
            raise ValueError(f'Incorrect parameters')

        # connect somehow with returing columns

    """@staticmethod
    def return_columns(data_set: pd.DataFrame(),
                       data_set_if_pre: pd.DataFrame() = None,
                       pre_split=False):

        if not pre_split:
            return list(data_set.columns[:-1])
        if pre_split:
            return {'columns_train': data_set.columns,
                    'columns_test': data_set_if_pre.columns}"""


class SubDataFrameGenerator:
    """
    Temp solution for high memory usage in brute force method
    """

    @staticmethod
    def generate_combinations(df):
        feature_columns = list(df.columns)
        feature_columns.remove('y')
        combinations = list(itertools.combinations(feature_columns, 5))
        combinations = [list(combination) + ['y'] for combination in combinations]

        return combinations

    @staticmethod
    def return_sub_df(df: pd.DataFrame(),
                      combination: list):
        df_5var = df[combination]
        return df_5var
