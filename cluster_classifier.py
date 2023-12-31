from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from joblib import dump
from os import makedirs, listdir
from os.path import join


def test_classifiers(x, y):
    """Try out a few different classifiers to examine their results"""

    classifiers = [
        SVC(kernel="linear", C=0.025, random_state=42),
        SVC(gamma=2, C=1, random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        MLPClassifier(max_iter=1000, random_state=42),
        QuadraticDiscriminantAnalysis(),
    ]

    scores = []

    for classifier in classifiers:
        classifier.fit(x, y)
        scores.append(classifier.score(x, y))

    return classifiers, scores


def classifier_grid_search(x, y):
    """Hypertune the classifier"""

    parameters = {
        "C": [0.025, 0.1, 1, 10, 100],
        "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
        "gamma": ["scale", "auto"],
        "kernel": ["linear"],
    }

    grid_search = GridSearchCV(SVC(), parameters)
    grid_search.fit(x, y)

    print(grid_search.best_params_)
    grid_predictions = grid_search.predict(x)

    print(classification_report(y, grid_predictions))
    print(confusion_matrix(y, grid_predictions))


def fit_classifier(x, y, output_file="trained_models/svm"):
    """Train a classifier"""
    svm = SVC(kernel="linear", C=0.1, random_state=42)
    svm.fit(x, y)
    training_accuracy = svm.score(x, y)
    dump(svm, output_file)
    return svm, training_accuracy


def create_groups(
    classifier,
    k: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    val_size: float = 0.1,
    output_dir="datasets/groups/",
):
    """
    Assigns data to clusters, and creates three partitions: train, val, and test.
    """

    # set up a list to store the subsets
    groups = []

    # run inference
    training_groups = classifier.predict(x_train)
    test_groups = classifier.predict(x_test)

    # create k subsets
    for i in range(k):
        # set aside a portion of the training data for validation
        xi_train, xi_val, yi_train, yi_val = train_test_split(
            x_train[training_groups == i],
            y_train[training_groups == i],
            test_size=val_size,
            random_state=1234,
        )

        group_data = {
            "x_train": xi_train,
            "y_train": yi_train,
            "x_val": xi_val,
            "y_val": yi_val,
            "x_test": x_test[test_groups == i],
            "y_test": y_test[test_groups == i],
        }

        groups.append(group_data)

        dir = join(output_dir, str(i))
        makedirs(dir, exist_ok=True)
        for key in group_data:
            np.save(join(dir, f"{key}.npy"), group_data[key])

    return groups


def assign_to_clusters(classifier, k, x, y) -> list[dict[str, np.ndarray]]:
    """
    Assigns data to clusters. No partitions, just x and y.
    """

    # set up a list to store the subsets
    groups = []

    # run inference
    group_ids = classifier.predict(x)

    # create k subsets
    for i in range(k):
        group_data = {
            "x": x[group_ids == i],
            "y": y[group_ids == i],
        }

        groups.append(group_data)

    return groups


def load_groups(group_dir) -> dict[str, np.ndarray]:
    data = {}
    for subset in listdir(group_dir):
        data[subset.removesuffix(".npy")] = np.load(join(group_dir, subset))
    return data


def save_cluster_to_disk(training_data: np.ndarray, training_clusters: np.ndarray):
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
