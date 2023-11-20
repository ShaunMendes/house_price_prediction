from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def preprocess(data: pd.date_range) -> tuple[pd.DataFrame, dict[str, LabelEncoder], StandardScaler]:
    '''
    TODO: finish doc string
    '''

    # TODO: reduce features from 79 to 35
    data = reduce_features(data)

    # TODO: handle remaining nans: remove/impute
    data = handle_nans(data)

    # convert categorical features to integers
    label_encoders = create_label_encoders(data)
    data = encode_categorical_features(data, label_encoders)

    # TODO: standardize/normalize data
    data, scaler = standardize_data(data)

    return data, label_encoders, scaler

def reduce_features(data: pd.DataFrame):
    ''' TODO: finish this implementation! '''

    data = data.drop(columns=[
        'PoolQC',       # 99.5% are nan.
        'MiscFeature',  # 96.9% are nan.
        'Fence',        # 80.6% are nan.
        'FireplaceQu',  # 47.0% are nan. Also highly correlated with 'Fireplaces'.
    ])

    return data

def handle_nans(data: pd.DataFrame) -> pd.DataFrame:
    # TODO: finish implementation!

    # drop rows containing nans. There are 54 rows with nans for 5 features related to garages.
    data = data.dropna()
    return data

def create_label_encoders(data: pd.DataFrame) -> dict[str, LabelEncoder]:
    '''
    Create a label encoder for each string feature in the provided dataset.
    Returns a dictionary of label encoders, which can be accessed using the
    name of the feature (pulled from the provided dataframe).
    '''
    label_encoders = {}
    for feature in data:
        if data[feature].dtype == 'object':
            label_encoders[feature] = LabelEncoder().fit(data[feature])
    return label_encoders

def encode_categorical_features(data: pd.DataFrame, label_encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    '''
    Use the provided label encoders to transform string features in the dataset to integers.
    Meant for use on both training and testing data.
    '''
    for feature in label_encoders:
        data[feature] = label_encoders[feature].transform(data[feature])
    return data

def standardize_data(data: pd.DataFrame) -> tuple[pd.DateOffset, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit_transform(data)
    return data, scaler