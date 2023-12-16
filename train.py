from group_models import TrainGroupModels
from joblib import dump, load
from preprocess import create_preprocessors, preprocess
from constants import *
from kmeans import train_kmeans
from cluster_classifier import fit_classifier, create_groups
import pandas as pd

def train(prefix=''):

    training_data = pd.read_csv("datasets/train.csv")
    test_data = pd.read_csv("datasets/test.csv")

    nan_replacements, label_encoders, feature_scaler, price_scaler, pca = create_preprocessors(training_data, drop_nans=DROP_TRAINING_NANS)

    x_train, y_train = preprocess(
        training_data,
        nan_replacements,
        label_encoders,
        feature_scaler,
        price_scaler,
        pca,
        drop_nans=DROP_TRAINING_NANS
    )

    x_test, y_test = preprocess(
        test_data,
        nan_replacements,
        label_encoders,
        feature_scaler,
        price_scaler,
        pca,
        drop_nans=False
    )

    # train kmeans and generate ground truth clusters
    kmeans = train_kmeans(x_train, K)
    training_clusters = kmeans.predict(x_train)
    test_clusters = kmeans.predict(x_test)

    # train the cluster classifier
    cluster_svm, training_accuracy = fit_classifier(x_train, training_clusters)
    test_accuracy = cluster_svm.score(x_test, test_clusters)
    print(f"\n--- Cluster Classifier ---\nTraining accuracy = {training_accuracy}\nTest Accuracy = {test_accuracy}\n")

    # use the classifier to split data into groups
    groups = create_groups(cluster_svm, K, x_train, y_train, x_test, y_test)
    # groups = create_groups(kmeans, K, x_train, y_train, x_test, y_test)

    # train group models
    trainer = TrainGroupModels(price_scaler=price_scaler, data_groups=groups)
    model0 = trainer.train_and_evaluate(group_id=0)
    model1 = trainer.train_and_evaluate(group_id=1)
    model2 = trainer.train_and_evaluate(group_id=2)

    # save trained models
    dump(model0, f"./trained_models/{prefix}model0.pkl")
    dump(model1, f"./trained_models/{prefix}model1.pkl")
    dump(model2, f"./trained_models/{prefix}model2.pkl")

def mass_training(prefix=''):
    for i in range(15):
        train(f'{prefix}_{i}_')    

if __name__ == "__main__":

    # train()
    mass_training('svc')