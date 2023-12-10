from preprocess import preprocess, create_preprocessors
from cluster_classifier import assign_to_clusters
from joblib import load
from constants import *
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def test_data_predicton(training_data: pd.DataFrame, test_data: pd.DataFrame):
    
    # use the training data to fit preprocessors
    (
        nan_replacements,
        label_encoders,
        feature_scaler,
        price_scaler,
        pca,
    ) = create_preprocessors(training_data, drop_nans=DROP_TRAINING_NANS)

    # preprocess the data
    x_test, y_test = preprocess(
        test_data,
        nan_replacements,
        label_encoders,
        feature_scaler,
        price_scaler,
        pca,
        drop_nans=False
    )

    # load trained classifier
    cluster_svm = load("trained_models/svm")

    # assign test samples to clusters using our classifier.
    # NOTE: the keys for the dicts stored in grouped_data list are 'x' and 'y'.
    grouped_data = assign_to_clusters(cluster_svm, K, x_test, y_test)

    # load trained group model
    models = {}
    models[0] = load("./trained_models/model0.pkl")
    models[1] = load("./trained_models/model1.pkl")
    models[2] = load("./trained_models/model2.pkl")

    # run inference
    y_preds = np.array([])
    y_true = np.array([])

    for i, group in enumerate(grouped_data):

        y_pred = models[i].predict(group["x"])

        y_true = np.append(y_true, group["y"])
        y_preds = np.append(y_preds, y_pred)

        mse = mean_squared_error(
            price_scaler.inverse_transform(group["y"].reshape(-1, 1)),
            price_scaler.inverse_transform(y_pred.reshape(-1, 1)),
        )

        print(f"The mse for group {i} is {mse} and rmse is {np.sqrt(mse)}")

    # compute mse
    y_test = price_scaler.inverse_transform(y_true.reshape(-1, 1))
    y_preds = price_scaler.inverse_transform(y_preds.reshape(-1, 1))
    mse = mean_squared_error(y_test, y_preds)

    print(f"The mse is {mse} and rmse is {np.sqrt(mse)}")

if __name__ == "__main__":
    training_data = pd.read_csv("datasets/train.csv")
    test_data = pd.read_csv("datasets/test.csv")
    test_data_predicton(training_data, test_data)