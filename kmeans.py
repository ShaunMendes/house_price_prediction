from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preproces import preprocess

def elbow_method(data, clusters_to_test):
    inertias = []
    if isinstance(clusters_to_test, int): clusters_to_test = list(range(1, clusters_to_test+1))
    for k in clusters_to_test:
        km = run_kmeans(data, k)
        inertias.append(km.inertia_)
    plt.plot(clusters_to_test, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

def run_kmeans(data, k, max_iter=300, num_trials=10):
    kmeans = KMeans(
        n_clusters=k,
        max_iter=max_iter,
        n_init=num_trials
    )
    kmeans.fit(data)
    return kmeans

if __name__ == '__main__':
    
    # TODO: replace with full dataset
    data = pd.read_csv('datasets/train_2.csv')

    # TODO: standardize data
    preprocessed_data, label_encoders = preprocess(data)

    elbow_method(preprocessed_data, 3)


