from preprocess import preprocess, create_preprocessors
from kmeans import train_kmeans
from cluster_classifier import (
    test_classifiers,
    fit_classifier,
    classifier_grid_search,
    create_groups,
    load_groups,
)
from joblib import dump
from group_models import TrainGroupModels, start_regressor_expeirments
import pandas as pd
import numpy as np
from os.path import exists
from constants import K

if __name__ == "__main__":

    training_data = pd.read_csv("datasets/train.csv")
    test_data = pd.read_csv("datasets/test.csv")

    nan_replacements, label_encoders, feature_scaler, price_scaler, pca = create_preprocessors(training_data)

    x_train, y_train = preprocess(
        training_data,
        nan_replacements,
        label_encoders,
        feature_scaler,
        price_scaler,
        pca
    )

    x_test, y_test = preprocess(
        test_data,
        nan_replacements,
        label_encoders,
        feature_scaler,
        price_scaler,
        pca
    )

    # train kmeans and generate ground truth clusters
    kmeans = train_kmeans(x_train, K)
    training_clusters = kmeans.predict(x_train)
    test_clusters = kmeans.predict(x_test)

    # train the cluster classifier
    cluster_svm, training_accuracy = fit_classifier(x_train, training_clusters)
    test_accuracy = cluster_svm.score(x_test, test_clusters)
    print(
        f"\n--- Cluster Classifier ---\nTraining accuracy = {training_accuracy}\nTest Accuracy = {test_accuracy}\n"
    )

    kmeans_groups = create_groups(kmeans, K, x_train, y_train, x_test, y_test)
    start_regressor_expeirments(kmeans_groups[0], price_scaler, 'kmeans_0')
    start_regressor_expeirments(kmeans_groups[1], price_scaler, 'kmeans_1')
    start_regressor_expeirments(kmeans_groups[2], price_scaler, 'kmeans_2')

    svc_groups = create_groups(cluster_svm, K, x_train, y_train, x_test, y_test)
    start_regressor_expeirments(svc_groups[0], price_scaler, 'svc_0')
    start_regressor_expeirments(svc_groups[1], price_scaler, 'svc_1')
    start_regressor_expeirments(svc_groups[2], price_scaler, 'svc_2')
