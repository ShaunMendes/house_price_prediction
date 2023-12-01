from group_models import TrainGroupModels
from joblib import dump, load
from preprocess import create_preprocessors, preprocess
from constants import K
from kmeans import train_kmeans
from cluster_classifier import fit_classifier, create_groups
import pandas as pd


if __name__ == '__main__':

    training_data = pd.read_csv('datasets/train.csv')
    test_data = pd.read_csv('datasets/test.csv')

    create_preprocessors(training_data)

    nan_replacements = load('trained_models/nan_replacements'),
    label_encoders = load('trained_models/label_encoders'),
    feature_scaler = load('trained_models/feature_scaler'),
    price_scaler = load('trained_models/price_scaler'),
    pca = load('trained_models/pca'),

    x_train, y_train = preprocess(
        training_data,
        nan_replacements,
        label_encoders,
        feature_scaler,
        price_scaler,
        pca
    )

    x_test, y_test = preprocess(
        training_data,
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
    print(f'\n--- Cluster Classifier ---\nTraining accuracy = {training_accuracy}\nTest Accuracy = {test_accuracy}\n')

    # use the classifier to split data into groups
    groups = create_groups(cluster_svm, K, x_train, y_train, x_test, y_test)    

    # train group models
    trainer = TrainGroupModels(price_scaler=price_scaler)
    model0, _, _ = trainer.train_and_evaluate(group_id=0)
    model1, _, _ = trainer.train_and_evaluate(group_id=1)
    model2, _, _ = trainer.train_and_evaluate(group_id=2)

    # TODO: save trained model, report results