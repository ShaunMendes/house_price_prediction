from preprocess import preprocess, create_preprocessors
from cluster_classifier import assign_to_clusters
from joblib import load
from constants import K
import pandas as pd
import numpy as np

def test_data_predicton(training_data: pd.DataFrame, test_data: pd.DataFrame):

    # use the training data to fit preprocessors
    nan_replacements, label_encoders, feature_scaler, price_scaler, pca = create_preprocessors(training_data)

    # preprocess the data
    x_test, y_test = preprocess(
        test_data,
        nan_replacements,
        label_encoders,
        feature_scaler,
        price_scaler,
        pca
    )

    # load trained classifier
    cluster_svm = load('trained_models/svm')

    # assign test samples to clusters using our classifier.
    # NOTE: the keys for the dicts stored in grouped_data list are 'x' and 'y'.
    grouped_data = assign_to_clusters(cluster_svm, K, x_test, y_test)

    # TODO: load trained group model

    # TODO: run inference

    # TODO compute mse

if __name__ == '__main__':

    training_data = pd.read_csv('datasets/train.csv')
    test_data = pd.read_csv('datasets/test.csv')

    test_data_predicton(training_data, test_data)