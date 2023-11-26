from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def test_classifiers(x, y):
    
    classifiers = [
        SVC(kernel="linear", C=0.025, random_state=42),
        SVC(gamma=2, C=1, random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        MLPClassifier(max_iter=1000, random_state=42),
        QuadraticDiscriminantAnalysis(),
    ]

    scores = []

    for classifier in classifiers:
        classifier.fit(x,y)
        scores.append(classifier.score(x,y))

    return classifiers, scores

def grid_search(x, y):
    parameters = {
        'C': [0.025, 0.1, 1, 10, 100],  
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
        'gamma':['scale', 'auto'],
        'kernel': ['linear']
    }
    grid_search = GridSearchCV(SVC(), parameters)
    grid_search.fit(x, y)

    print(grid_search.best_params_) 
    grid_predictions = grid_search.predict(x) 
    
    print(classification_report(y, grid_predictions)) 
    print(confusion_matrix(y, grid_predictions))

def fit_classifier(x, y):
    # Try a linear classifier first. if linear models don't work well enough, then try a non linear
    svm = SVC(kernel="linear", C=0.1, random_state=42)
    svm.fit(x, y)
    training_accuracy = svm.score(x, y)
    return svm, training_accuracy


def save_cluster_to_disk(training_data: np.array, training_clusters: np.array):
    """
    Saves each cluster numpy array onto disk without cluster id
    """

    clusters = np.unique(training_clusters)
    training_clusters = np.expand_dims(training_clusters, axis=1)
    training_clusters_w_cluster_id = np.concatenate(
        (training_data, training_clusters), axis=1
    )
    for cluster in clusters:
        with open(f"datasets/cluster_{cluster}.npy", "wb") as f:
            np.save(
                f,
                training_clusters_w_cluster_id[
                    training_clusters_w_cluster_id[:, -1] == cluster
                ][:, :-1],
            )
