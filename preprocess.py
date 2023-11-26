from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def preprocess(training_data: pd.DataFrame, test_data: pd.DataFrame, standardize_price: bool = True) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, LabelEncoder], StandardScaler, PCA]:
    
    '''
    A single function to preprocess training and testing data.
    - A list of features will be dropped based on nan count, correlation to other features, etc...
    - The emaining nans are removed/imputed.
    - All categorical features are converted to integers.
    - The features are then standardized.
    - PCA is used to reduce the number of features to 35.
    '''

    # Use feature selection to reduce feature count
    training_data = reduce_features(training_data)
    test_data = reduce_features(test_data)

    # handle remaining nans: remove/impute
    training_data = handle_nans(training_data)
    test_data = handle_nans(test_data)

    # convert categorical features to integers
    label_encoders = create_label_encoders(training_data)
    training_data = encode_categorical_features(training_data, label_encoders)
    test_data = encode_categorical_features(test_data, label_encoders)

    # Seperate the input features from the target and covnert to numpy arrays
    x_train = training_data.drop(columns=['SalePrice']).to_numpy()
    y_train = training_data['SalePrice'].to_numpy()
    x_test = test_data.drop(columns=['SalePrice']).to_numpy()
    y_test = test_data['SalePrice'].to_numpy()

    # standardize features
    if standardize_price:
        x_train, x_test, y_train, y_test, scaler = standardize_data(x_train, x_test, y_train, y_test)
    else:
        x_train, x_test, scaler = standardize_data(x_train, x_test)

    # utilize pca to drop remaining feature count to 35
    x_train, x_test, pca = run_pca(x_train, x_test)

    return x_train, y_train, x_test, y_test, label_encoders, scaler, pca

def reduce_features(data: pd.DataFrame):
    ''' 
    Drop features that have many nan values, are highly correlated with other features, 
    or that are largely uniform.
    '''

    data = data.drop(columns=[
        'PoolQC',       # 99.5% are nan.
        'MiscFeature',  # 96.9% are nan.
        'Fence',        # 80.6% are nan.
        'FireplaceQu',  # 47.0% are nan. Also highly correlated with 'Fireplaces'.
        'GarageCars',   # Highly correlated with 'GarageArea'.
        'TotRmsAbvGrd', # Highly correlated with 'GrLivArea'.
        'TotalBsmtSF',  # Highly correlated with '1stFlrSF'.
        'Exterior2nd',  # Highly correlated with 'Exterior1st'.
        'GarageYrBlt',  # Highly correlated with 'GarageFinish'.
        'BldgType',     # Highly correlated with 'MSSubClass', and many values in one category.
        'LandSlope',    # Highly correlated with 'LandContour, and many values in one category.
        'LowQualFinSF', # Majority of values belong to a single category.
        'KitchenAbvGr', # Majority of values belong to a single category.
        'Heating',      # Majority of values belong to a single category.
        'YearBuilt',    # Highly correlated with many other features.
        'OverallQual',  # Highly correlated with many other features.

        'GarageCond',   # 

        'Neighborhood', # TODO: test data has categories not found in training data
        'Exterior1st',  # TODO: test data has categories not found in training data... weak correlation with price
        'Functional',   # TODO: test data has categories not found in training data... weak correlation with price


        "Unnamed: 0",   # just row numbers
        "Id",           # has all unique ids
        "Street",       # Majority values belong to single category
        "MasVnrType",   # Majority of values are nan
        "Alley",        # Majority of values are nan
        "Utilities",    # Majority values belong to single category
        "Condition2",   # Majority values belong to single category
        "RoofMatl",     # Majority values belong to single category
        "MasVnrArea",   # Majority values belong to single category - can be added back later
        "BsmtFinSF1",   # Majority values belong to single category
        "BsmtFinSF2"    # Majority values belong to single category
    ])

    return data

def handle_nans(data: pd.DataFrame) -> pd.DataFrame:
    ''' TODO: finish implementation & doc string '''

    data["LotFrontage"] = data["LotFrontage"].fillna(data["LotFrontage"].mean())

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

def standardize_data(x_train, x_test, y_train=None, y_test=None):
    ''' TODO: finish doc string '''
    scaler = StandardScaler()
    if y_train is None or y_test is None:
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        return x_train, x_test, scaler
    else: 
        train = np.column_stack([x_train, y_train])
        test = np.column_stack([x_test, y_test])
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
        x_train, x_test, y_train, y_test = train[:, :-1], test[:, :-1], train[:, -1], test[:, -1]
        return x_train, x_test, y_train, y_test, scaler

def run_pca(training_data: np.ndarray, test_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, PCA]:
    ''' Use PCA on the training and testing data to reduce the feature count. '''
    pca=PCA(n_components=35)
    training_data = pca.fit_transform(training_data)
    test_data = pca.transform(test_data)
    return training_data, test_data, pca