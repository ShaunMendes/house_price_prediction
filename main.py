from preprocess import preprocess, preprocess_for_inference
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

K = 3

def test_data_predicton(training_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    TODO: this is a function the professor requires us to implement. requirements listed below:

    Write a function, test_data_prediction(train data, test data), that transforms the test
    data into the train data format, e.g., the same features, scales, etc, and predicts the
    house price using the trained best models.

    NOTE: does the prof expect us to run training and test inference from this function?
        if not, then why accept the training data as an input?

    """

    # preprocess data
    x_train, y_train, x_test, y_test, label_encoders, scaler, pca = preprocess(
        training_data, test_data, standardize_price=True
    )

    """
    TODO: train the models here? load prexisting trained models? pass the models in?

    - use the classifier to predict which group each sample belongs to
    - use the group model that corresponds to each sample's predicted group to run inference.
    - return results?

    """


if __name__ == "__main__":
    # # load and preprocess data
    # training_data = pd.read_csv('datasets/train.csv')
    test_data = pd.read_csv('datasets/test.csv')
    # x_train, y_train, x_test, y_test, label_encoders, feature_scaler, price_scaler, pca = preprocess(training_data, test_data, standardize_price=True)
    preprocess_for_inference(
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
