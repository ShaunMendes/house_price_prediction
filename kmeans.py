from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from joblib import dump

def elbow_method(data, clusters_to_test):
    inertias = []
    if isinstance(clusters_to_test, int): clusters_to_test = list(range(1, clusters_to_test+1))
    for k in clusters_to_test:
        km = train_kmeans(data, k)
        inertias.append(km.inertia_)
    plt.plot(clusters_to_test, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig('ElbowMethodGraph.png')
    plt.show()
    
def train_kmeans(data, k, max_iter=300, num_trials=10, output_file='trained_models/kmeans'):
    kmeans = KMeans(
        n_clusters=k,
        max_iter=max_iter,
        n_init=num_trials,
        random_state=1234
    )
    kmeans.fit(data)
    dump(kmeans, output_file)
    return kmeans

def load_kmeans():
    pass