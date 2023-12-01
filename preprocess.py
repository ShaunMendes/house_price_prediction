from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from joblib import dump
from os.path import join
from os import makedirs
from typing import Union


def create_preprocessors(
    training_data: pd.DataFrame,
    standardize_price: bool = True,
    model_dir="trained_models",
) -> tuple[
    dict,
    dict[str, LabelEncoder],
    dict[str, str],
    StandardScaler,
    StandardScaler,
    PCA,
]:
    """
    Create several sklearn objects to preprocess the data
    - a dict of label encoders to convert categorical features to integers
    - a feature scaler to standardize input features
    - a price scaler to standardize the target variable
    - a pca instance used to drop the feature count to 35
    """

    # Use feature selection to reduce feature count
    training_data = reduce_features(training_data)

    # handle remaining nans: remove/impute
    training_nan_replacements = create_nan_replacements(training_data)
    training_data = fill_nans(training_data, training_nan_replacements)

    # convert categorical features to integers
    label_encoders, most_common_label = create_label_encoders(training_data)
    training_data = encode_categorical_features(
        training_data, label_encoders, most_common_label
    )

    # Seperate the input features from the target and covnert to numpy arrays
    x_train = training_data.drop(columns=["SalePrice"]).to_numpy()
    y_train = training_data["SalePrice"].to_numpy()

    # create and fit a scaler for the input features
    feature_scaler = StandardScaler()
    feature_scaler.fit(x_train)
    x_train = standardize_data(x_train, feature_scaler)

    # standardize price if necessary
    if standardize_price:
        price_scaler = StandardScaler()
        price_scaler.fit(y_train.reshape(-1, 1))
        y_train = standardize_data(y_train, price_scaler)

    # utilize pca to drop remaining feature count to 35
    x_train, pca = run_pca(x_train)

    # save all trained components
    makedirs(model_dir, exist_ok=True)
    dump(training_nan_replacements, join(model_dir, "nan_replacements"))
    dump(label_encoders, join(model_dir, "label_encoders"))
    dump(most_common_label, join(model_dir, "most_common_label"))
    dump(feature_scaler, join(model_dir, "feature_scaler"))
    dump(price_scaler, join(model_dir, "price_scaler"))
    dump(pca, join(model_dir, "pca"))

    if standardize_price:
        return (
            training_nan_replacements,
            label_encoders,
            most_common_label,
            feature_scaler,
            price_scaler,
            pca,
        )
    else:
        return (
            training_nan_replacements,
            label_encoders,
            most_common_label,
            feature_scaler,
            pca,
        )


def preprocess(
    data: pd.DataFrame,
    nan_replacements: dict[str, float],
    label_encoders: dict[str, LabelEncoder],
    most_common_label: dict[str, str],
    feature_scaler: StandardScaler,
    price_scaler: StandardScaler | None,
    pca: PCA,
) -> tuple[np.ndarray, np.ndarray]:
    """Utilized pre-generated preprocessors to prepare the data"""

    data = reduce_features(data)
    data = fill_nans(data, nan_replacements)
    data = encode_categorical_features(data, label_encoders, most_common_label)

    x = data.drop(columns=["SalePrice"]).to_numpy()
    y = data["SalePrice"].to_numpy()

    x = standardize_data(x, feature_scaler)
    if price_scaler is not None:
        y = standardize_data(y, price_scaler)

    x = pca.transform(x)

    return x, y


def reduce_features(data: pd.DataFrame):
    """
    Drop features that have many nan values, are highly correlated with other features,
    or that are largely uniform.
    """

    data = data.drop(
        columns=[
            "PoolQC",  # 99.5% are nan.
            "MiscFeature",  # 96.9% are nan.
            "Fence",  # 80.6% are nan.
            "FireplaceQu",  # 47.0% are nan. Also highly correlated with 'Fireplaces'.
            # 'TotRmsAbvGrd', # Highly correlated with 'GrLivArea'.
            "Exterior2nd",  # Highly correlated with 'Exterior1st'.
            "GarageFinish",  # Highly correlated with 'GarageYrBlt'.
            "BldgType",  # Highly correlated with 'MSSubClass', and many values in one category.
            "LandSlope",  # Highly correlated with 'LandContour, and many values in one category.
            "LowQualFinSF",  # Majority of values belong to a single category.
            "KitchenAbvGr",  # Majority of values belong to a single category.
            "Heating",  # Majority of values belong to a single category.
            # 'YearBuilt',    # Highly correlated with many other features.
            # 'OverallQual',  # Highly correlated with many other features.
            "1stFlrSF",  # rolled into GrLivArea
            "2ndFlrSF",  # rolled into GrLivArea
            "GarageCond",  #
            "GarageYrBlt",  # correlated with 'YearBuilt'.
            "GarageArea",  # correlated with 'GarageCars'.
            "Foundation",  # correlated with 'YearBuilt'.
            "Neighborhood",  # TODO: test data has categories not found in training data
            "Exterior1st",  # TODO: test data has categories not found in training data... weak correlation with price
            "Functional",  # TODO: test data has categories not found in training data... weak correlation with price
            "Unnamed: 0",  # just row numbers
            "Id",  # has all unique ids
            "Street",  # Majority values belong to single category
            "MasVnrType",  # Majority of values are nan
            "Alley",  # Majority of values are nan
            "Utilities",  # Majority values belong to single category
            "Condition2",  # Majority values belong to single category
            "RoofMatl",  # Majority values belong to single category
            "MasVnrArea",  # Majority values belong to single category - can be added back later
            "BsmtFinSF1",  # Majority values belong to single category. apart of TotalBsmtSF
            "BsmtFinSF2",  # Majority values belong to single category. apart of TotalBsmtSF
            "BsmtUnfSF",  # apart of TotalBsmtSF.
        ]
    )

    return data


def create_nan_replacements(data: pd.DataFrame) -> dict:
    """
    loop through the provided dataset and search for features with nans.
    for each one of these features, determine a value that would be a suitable replacement for nans.
    for numeric values, the average will be used.
    for categorical values, the most common value will be used.
    """

    nan_replacements = {}

    for feature in data:
        if data[feature].isna().sum() > 0:
            if data[feature].dtype == "object":
                nan_replacements[feature] = data[feature].value_counts().idxmax()
            else:
                nan_replacements[feature] = data[feature].mean()

    return nan_replacements


def fill_nans(data: pd.DataFrame, nan_replacements: dict[str, float]) -> pd.DataFrame:
    """use the nan replacement dict to replace nans within the dataset"""
    for feature in nan_replacements:
        data[feature] = data[feature].fillna(nan_replacements[feature])
    return data


def create_label_encoders(
    data: pd.DataFrame,
) -> (dict[str, LabelEncoder], dict[str, str]):
    """
    Create a label encoder for each string feature in the provided dataset.
    Returns a dictionary of label encoders, which can be accessed using the
    name of the feature (pulled from the provided dataframe).
    """
    label_encoders = {}
    most_common_label = {}
    for feature in data:
        if data[feature].dtype == "object":
            most_common_label[feature] = data[feature].mode().iloc[0]
            data[feature] = data[feature].fillna(most_common_label[feature])
            label_encoders[feature] = LabelEncoder().fit(data[feature])
    return label_encoders, most_common_label


def encode_categorical_features(
    data: pd.DataFrame,
    label_encoders: dict[str, LabelEncoder],
    most_common_label: dict[str, str],
) -> pd.DataFrame:
    """
    Use the provided label encoders to transform string features in the dataset to integers.
    Meant for use on both training and testing data.
    """
    for feature in label_encoders:
        data[feature] = data[feature].fillna(most_common_label[feature])
        data[feature] = label_encoders[feature].transform(data[feature])
    return data


def standardize_data(data: np.ndarray, scaler) -> np.ndarray:
    """Use the provided StandardScaler to standardize the numpy array"""
    if len(data.shape) == 1:
        data = data.reshape([-1, 1])
    data = scaler.transform(data)
    return data


def run_pca(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Use PCA on the training and testing data to reduce the feature count."""
    pca = PCA(n_components=35, random_state=1234)
    training_data = pca.fit_transform(data)
    # test_data = pca.transform(test_data)
    return training_data, pca
