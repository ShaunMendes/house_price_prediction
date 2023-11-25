from preprocess import preprocess
from kmeans import elbow_method, run_kmeans
from cluster_classifier import test_classifiers, fit_classifier, grid_search
import pandas as pd

K = 4

if __name__ == '__main__':
    
    training_data = pd.read_csv('datasets/train.csv')
    test_data = pd.read_csv('datasets/test.csv')
    x_train, y_train, x_test, y_test, label_encoders, scaler, pca = preprocess(training_data, test_data)

    kmeans = run_kmeans(x_train, K)
    training_clusters = kmeans.predict(x_train)

    '''
    TODO:
        - split training data by cluster... use kmeans or classifier to do this? im assuming its the classifier
        - create validation sets from the training data
    '''

    # classifiers, scores = test_classifiers(training_data, training_clusters)
    grid_search(training_data, training_clusters)
    # cluster_svm, training_accuracy = fit_classifier(training_data, training_clusters)

    # TODO: train group regression models

    # TODO: create inference function that uses the trained models to make predictions on the test data

    pass