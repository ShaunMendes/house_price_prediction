from preprocess import preprocess
from kmeans import elbow_method, train_kmeans
from cluster_classifier import (
    test_classifiers,
    fit_classifier,
    classifier_grid_search,
    create_groups,
    load_groups,
)
from joblib import dump, load
from group_models import TrainGroupModels, start_regressor_expeirments
import pandas as pd
import numpy as np
from os.path import exists
from constants import K

if __name__ == "__main__":
    # # load and preprocess data
    # training_data = pd.read_csv('datasets/train.csv')
    test_data = pd.read_csv('datasets/test.csv')
    # x_train, y_train, x_test, y_test, label_encoders, feature_scaler, price_scaler, pca = preprocess(training_data, test_data, standardize_price=True)
    x_test, y_test = preprocess(
        test_data,
        load('trained_models/nan_replacements'),
        load('trained_models/label_encoders'),
        load('trained_models/feature_scaler'),
        load('trained_models/price_scaler'),
        load('trained_models/pca'),
    )


    # load price scaler
    price_scaler = load('trained_models/price_scaler')

    # # train kmeans and generate ground truth clusters
    # kmeans = train_kmeans(x_train, K)
    # training_clusters = kmeans.predict(x_train)
    # test_clusters = kmeans.predict(x_test)

    # load trained kmeans
    kmeans = load('trained_models/kmeans')

    # # determine which the best classifier and its hyperparameters.
    # classifiers, scores = test_classifiers(x_train, training_clusters)
    # grid_search(x_train, training_clusters)

    # # train the cluster classifier
    # cluster_svm, training_accuracy = fit_classifier(x_train, training_clusters)
    # test_accuracy = cluster_svm.score(x_test, test_clusters)
    # print(f'\n--- Cluster Classifier ---\nTraining accuracy = {training_accuracy}\nTest Accuracy = {test_accuracy}\n')

    # load trained svm
    cluster_svm = load('trained_models/svm')
    # print(cluster_svm.score(x_test, kmeans.predict(x_test)))

    # use the classifier to split data into groups
    # groups = create_groups(cluster_svm, K, x_train, y_train, x_test, y_test)

    # hypertune model
    # start_regressor_expeirments(load_groups('datasets/groups/0'), price_scaler, 'rev4')

    # TODO: train group models
    trainer = TrainGroupModels(price_scaler=price_scaler)
    model0, _, _ = trainer.train_and_evaluate(group_id=0)
    model1, _, _ = trainer.train_and_evaluate(group_id=1)
    model2, _, _ = trainer.train_and_evaluate(group_id=2)

    pass
