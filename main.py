from preprocess import preprocess
from kmeans import elbow_method, run_kmeans
from cluster_classifier import test_classifiers, fit_classifier, grid_search
from sklearn.model_selection import train_test_split
from pickle import dump, load
import pandas as pd
import numpy as np

K = 4

def create_subsets(classifier, k: int, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, val_size: float = 0.1):

    # set up a list to store the subsets
    subsets = []

    # run inference
    training_groups = classifier.predict(x_train)
    test_groups = classifier.predict(x_test)

    # set up a list for sanity check TODO: remove this
    train_sanity_check = 0
    test_sanity_check = 0

    # create k subsets
    for i in range(k):

        # set aside a portion of the training data for validation
        xi_train, xi_val, yi_train, yi_val = train_test_split(
            x_train[training_groups == i], 
            y_train[training_groups == i], 
            test_size=val_size, 
            random_state=1234
        )

        subsets.append({

            'x_train': xi_train,
            'y_train': yi_train,

            'x_val': xi_val,
            'y_val': yi_val,

            'x_test': x_test[test_groups == i],
            'y_test': y_test[test_groups == i],

        })

        train_sanity_check += len(xi_train) + len(xi_val)
        test_sanity_check += len(x_test[test_groups == i])

    assert len(x_train) == train_sanity_check
    assert len(x_test) == test_sanity_check

    return subsets

if __name__ == '__main__':
    
    training_data = pd.read_csv('datasets/train.csv')
    test_data = pd.read_csv('datasets/test.csv')
    x_train, y_train, x_test, y_test, label_encoders, scaler, pca = preprocess(training_data, test_data)

    kmeans = run_kmeans(x_train, K)
    training_clusters = kmeans.predict(x_train)
    test_clusters = kmeans.predict(x_test)

    '''
    TODO:
        - split training data by cluster... use kmeans or classifier to do this? im assuming its the classifier
        - create validation sets from the training data
    '''

    # # determine which the best classifier and its hyperparameters.
    # classifiers, scores = test_classifiers(x_train, training_clusters)
    # grid_search(x_train, training_clusters)

    # train the cluster classifier
    cluster_svm, training_accuracy = fit_classifier(x_train, training_clusters)
    test_accuracy = cluster_svm.score(x_test, test_clusters)
    print(f'\n--- Cluster Classifier ---\nTraining accuracy = {training_accuracy}\nTest Accuracy = {test_accuracy}\n')

    # use the classifier to split data into groups
    subsets = create_subsets(cluster_svm, K, x_train, y_train, x_test, y_test)

    # code to dump the subsets to a pickle file
    subset_file = open('subsets.pkl', 'wb')
    dump(subsets, subset_file)
    subset_file.close()

    # TODO: train group regression models

    # TODO: create inference function that uses the trained models to make predictions on the test data

    pass