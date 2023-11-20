from preprocess import preprocess
from kmeans import elbow_method, run_kmeans
from cluster_classifier import test_classifiers, fit_classifier
import pandas as pd

K = 4

if __name__ == '__main__':
    
    training_data = pd.read_csv('datasets/train_2.csv')
    training_data, label_encoders, scaler = preprocess(training_data)

    kmeans = run_kmeans(training_data, K)
    training_clusters = kmeans.predict(training_data)

    # classifiers, scores = test_classifiers(training_data, training_clusters)
    cluster_svm, training_accuracy = fit_classifier(training_data, training_clusters)

    pass