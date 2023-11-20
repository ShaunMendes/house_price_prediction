from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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


def fit_classifier(x, y):
    # Try a linear classifier first. if linear models don't work well enough, then try a non linear
    svm = SVC(kernel="linear", C=0.025, random_state=42)
    svm.fit(x, y)
    training_accuracy = svm.score(x, y)
    return svm, training_accuracy
