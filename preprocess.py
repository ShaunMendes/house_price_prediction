from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from joblib import dump
from os.path import join
from os import makedirs


def preprocess(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    standardize_price: bool = True,
    model_dir="trained_models",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, LabelEncoder],
    StandardScaler,
    StandardScaler,
    PCA,
]:
    """
    A single function to preprocess training and testing data.
    - A list of features will be dropped based on nan count, correlation to other features, etc...
    - The emaining nans are removed/imputed.
    - All categorical features are converted to integers.
    - The features are then standardized.
    - PCA is used to reduce the number of features to 35.
    """

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
    x_train = training_data.drop(columns=["SalePrice"]).to_numpy()
    y_train = training_data["SalePrice"].to_numpy()
    x_test = test_data.drop(columns=["SalePrice"]).to_numpy()
    y_test = test_data["SalePrice"].to_numpy()

    # standardize features
    if standardize_price:
        x_train, x_test, feature_scaler = standardize_data(x_train, x_test)
        y_train, y_test, price_scaler = standardize_data(y_train, y_test)
    else:
        x_train, x_test, feature_scaler = standardize_data(x_train, x_test)

    # utilize pca to drop remaining feature count to 35
    x_train, x_test, pca = run_pca(x_train, x_test)

    # save all trained components
    makedirs(model_dir, exist_ok=True)
    dump(label_encoders, join(model_dir, "label_encoders"))
    dump(feature_scaler, join(model_dir, "feature_scaler"))
    dump(price_scaler, join(model_dir, "price_scaler"))
    dump(pca, join(model_dir, "pca"))

    if standardize_price:
        return (
            x_train,
            y_train.flatten(),
            x_test,
            y_test.flatten(),
            label_encoders,
            feature_scaler,
            price_scaler,
            pca,
        )
    else:
        return x_train, y_train, x_test, y_test, label_encoders, feature_scaler, pca


def preprocess_for_inference(
    test_data: pd.DataFrame,
    label_encoders: dict[str, LabelEncoder],
    feature_scaler: StandardScaler,
    price_scaler: StandardScaler,
    pca: PCA,
):
    test_data = reduce_features(test_data)
    test_data = handle_nans(test_data)
    test_data = encode_categorical_features(test_data, label_encoders)

    x_test = test_data.drop(columns=["SalePrice"]).to_numpy()
    y_test = test_data["SalePrice"].to_numpy()

    x_test = feature_scaler.transform(x_test)
    y_test = price_scaler.transform(y_test)
    x_test = pca.transform(x_test)

    return x_test, y_test


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


def handle_nans(
    data: pd.DataFrame, save_path: str = "", is_test: bool = False
) -> pd.DataFrame:
    """TODO: finish implementation & doc string"""

    data["LotFrontage"] = data["LotFrontage"].fillna(data["LotFrontage"].mean())

    # drop rows containing nans. There are 54 rows with nans for 5 features related to garages.
    data = data.dropna()
    return data


def create_label_encoders(data: pd.DataFrame) -> dict[str, LabelEncoder]:
    """
    Create a label encoder for each string feature in the provided dataset.
    Returns a dictionary of label encoders, which can be accessed using the
    name of the feature (pulled from the provided dataframe).
    """
    label_encoders = {}
    for feature in data:
        if data[feature].dtype == "object":
            label_encoders[feature] = LabelEncoder().fit(data[feature])
    return label_encoders


def encode_categorical_features(
    data: pd.DataFrame, label_encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:
    """
    Use the provided label encoders to transform string features in the dataset to integers.
    Meant for use on both training and testing data.
    """
    for feature in label_encoders:
        data[feature] = label_encoders[feature].transform(data[feature])
    return data


def standardize_data(train: np.ndarray, test: np.ndarray):
    """TODO: finish doc string"""
    scaler = StandardScaler()
    if len(train.shape) == 1:
        train = train.reshape([-1, 1])
    if len(test.shape) == 1:
        test = test.reshape([-1, 1])
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test, scaler


def run_pca(
    training_data: np.ndarray, test_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Use PCA on the training and testing data to reduce the feature count."""
    pca = PCA(n_components=35, random_state=1234)
    training_data = pca.fit_transform(training_data)
    test_data = pca.transform(test_data)
    return training_data, test_data, pca
