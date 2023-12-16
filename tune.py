import optuna
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from preprocess import preprocess, create_preprocessors
from kmeans import train_kmeans
from cluster_classifier import fit_classifier, create_groups
from constants import K

def tune_lr(data: dict[str, np.ndarray], scaler, db_name="db"):
    def regressor_trial(trial: optuna.Trial):
        regressor = LinearRegression(
            fit_intercept=trial.suggest_categorical("lr_fit_intercept", [True, False]),
            positive=trial.suggest_categorical("lr_positive", [True, False]),
        )

        regressor.fit(data["x_train"], data["y_train"])
        return np.sqrt(mean_squared_error(
            scaler.inverse_transform(data["y_val"].reshape(-1, 1)),
            scaler.inverse_transform(regressor.predict(data["x_val"]).reshape(-1, 1)),
        ))

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{db_name}.sqlite3",  # Specify the storage URL here.
        study_name="lr",
        load_if_exists=True,
    )

    study.optimize(regressor_trial, n_trials=4)

def tune_lasso(data: dict[str, np.ndarray], scaler, db_name="db"):
    def regressor_trial(trial: optuna.Trial):
        regressor = Lasso(
            selection=trial.suggest_categorical("lasso_selection", ["cyclic", "random"])
        )

        regressor.fit(data["x_train"], data["y_train"])
        return np.sqrt(mean_squared_error(
            scaler.inverse_transform(data["y_val"].reshape(-1, 1)),
            scaler.inverse_transform(regressor.predict(data["x_val"]).reshape(-1, 1)),
        ))

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{db_name}.sqlite3",
        study_name="lasso",
        load_if_exists=True,
    )

    study.optimize(regressor_trial, n_trials=2)

def tune_ridge(data: dict[str, np.ndarray], scaler, db_name="db"):
    def regressor_trial(trial: optuna.Trial):
        regressor = Ridge(
            alpha=trial.suggest_float("ridge_alpha", 0, 1000, step=0.01),
            solver=trial.suggest_categorical(
                "ridge_solver",
                ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            ),
        )

        regressor.fit(data["x_train"], data["y_train"])
        return np.sqrt(mean_squared_error(
            scaler.inverse_transform(data["y_val"].reshape(-1, 1)),
            scaler.inverse_transform(regressor.predict(data["x_val"]).reshape(-1, 1)),
        ))

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{db_name}.sqlite3",
        study_name="ridge",
        load_if_exists=True,
    )

    study.optimize(regressor_trial, n_trials=200)

def tune_svr(data: dict[str, np.ndarray], scaler, db_name="db"):
    def regressor_trial(trial: optuna.Trial):
        regressor = SVR(
            kernel=trial.suggest_categorical(
                "svr_kernel", ["linear", "poly", "rbf", "sigmoid"]
            ),
            degree=trial.suggest_int("svr_degree", 1, 5),
            gamma=trial.suggest_categorical("svr_gamma", ["scale", "auto"]),
            C=trial.suggest_float("svr_C", 1e-7, 10, log=True),
        )

        regressor.fit(data["x_train"], data["y_train"].flatten())
        return np.sqrt(mean_squared_error(
            scaler.inverse_transform(data["y_val"].reshape(-1, 1)),
            scaler.inverse_transform(regressor.predict(data["x_val"]).reshape(-1, 1)),
        ))

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{db_name}.sqlite3",
        study_name="svr",
        load_if_exists=True,
    )

    study.optimize(regressor_trial, n_trials=250)

def tune_decision_tree(data: dict[str, np.ndarray], scaler, db_name="db"):
    def regressor_trial(trial: optuna.Trial):
        regressor = DecisionTreeRegressor(
            splitter=trial.suggest_categorical("dt_splitter", ["best", "random"])
        )

        regressor.fit(data["x_train"], data["y_train"])
        return np.sqrt(mean_squared_error(
            scaler.inverse_transform(data["y_val"].reshape(-1, 1)),
            scaler.inverse_transform(regressor.predict(data["x_val"]).reshape(-1, 1)),
        ))

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{db_name}.sqlite3",
        study_name="dt",
        load_if_exists=True,
    )

    study.optimize(regressor_trial, n_trials=2)

def tune_mlrp(data: dict[str, np.ndarray], scaler, db_name="db"):
    def regressor_trial(trial: optuna.Trial):
        layers = []
        for layer in range(trial.suggest_int("mlpr_num_layers", 1, 3)):
            layers.append(trial.suggest_int(f"mlpr_{layer}_neurons", 5, 75, step=5))

        regressor = MLPRegressor(
            hidden_layer_sizes=layers,
            activation=trial.suggest_categorical(
                "mlpr_activation", ["relu", "identity", "logistic", "tanh"]
            ),
            solver=trial.suggest_categorical("mlpr_solver", ["sgd", "adam"]),
            alpha=trial.suggest_float("mlrp_alpha", 1e-7, 1e-1, log=True),
            learning_rate=trial.suggest_categorical(
                "mlrp_lr", ["constant", "invscaling", "adaptive"]
            ),
            learning_rate_init=trial.suggest_float(
                "mlrp_lr_init", 1e-7, 1e-1, log=True
            ),
        )

        try:
            regressor.fit(data["x_train"], data["y_train"].flatten())
            return np.sqrt(mean_squared_error(
                scaler.inverse_transform(data["y_val"].reshape(-1, 1)),
                scaler.inverse_transform(regressor.predict(data["x_val"]).reshape(-1, 1)),
            ))
        except:
            # an exception is thrown if the weights or predictions become infinite
            return None

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{db_name}.sqlite3",
        study_name="mlpr",
        load_if_exists=True,
    )

    study.optimize(regressor_trial, n_trials=300)

def tune_boosting(data: dict[str, np.ndarray], scaler, db_name="db"):

    def regressor_trial(trial: optuna.Trial):
        regressor = GradientBoostingRegressor(
        loss=trial.suggest_categorical('boost_loss', ['squared_error', 'absolute_error', 'huber', 'quantile']),
        learning_rate=trial.suggest_float("boost_lr_init", 1e-7, 1e-1, log=True),
        n_estimators=trial.suggest_int('boost_n_estimators', 10, 1000, step=10),
        subsample=trial.suggest_float('subsample', 0.1, 0.9, step=0.1),
        random_state=1234,
        alpha=trial.suggest_float('boost_alpha', 0.05, 0.95, step=0.05),
        warm_start=trial.suggest_categorical('boost_warm_start', [True, False]),
        )

        regressor.fit(data["x_train"], data["y_train"].flatten())
        return np.sqrt(mean_squared_error(
            scaler.inverse_transform(data["y_val"].reshape(-1, 1)),
            scaler.inverse_transform(regressor.predict(data["x_val"]).reshape(-1, 1)),
        ))

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{db_name}.sqlite3",
        study_name="boosting",
        load_if_exists=True,
    )

    study.optimize(regressor_trial, n_trials=200)

def start_regressor_expeirments(data, price_scaler, db_name):
    """
    Hypertune each type of model listed below.
    Report the results and the index of the best model.
    """

    tune_lr(data, price_scaler, db_name)
    tune_lasso(data, price_scaler, db_name)
    tune_ridge(data, price_scaler, db_name)
    tune_svr(data, price_scaler, db_name)
    tune_decision_tree(data, price_scaler, db_name)
    tune_mlrp(data, price_scaler, db_name)
    tune_boosting(data, price_scaler, db_name)

    scores = []
    for study in optuna.study.get_all_study_summaries(
        storage=f"sqlite:///{db_name}.sqlite3"
    ):
        print(study.best_trial.params)
        print(study.best_trial.value, end="\n\n")
        scores.append(study.best_trial.value)
    print(f"best index = {np.array(scores).argmin()}")

if __name__ == "__main__":

    training_data = pd.read_csv("datasets/train.csv")
    test_data = pd.read_csv("datasets/test.csv")

    nan_replacements, label_encoders, feature_scaler, price_scaler, pca = create_preprocessors(training_data)

    x_train, y_train = preprocess(
        training_data,
        nan_replacements,
        label_encoders,
        feature_scaler,
        price_scaler,
        pca
    )

    x_test, y_test = preprocess(
        test_data,
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
    print(
        f"\n--- Cluster Classifier ---\nTraining accuracy = {training_accuracy}\nTest Accuracy = {test_accuracy}\n"
    )

    kmeans_groups = create_groups(kmeans, K, x_train, y_train, x_test, y_test)
    start_regressor_expeirments(kmeans_groups[0], price_scaler, 'kmeans_0')
    start_regressor_expeirments(kmeans_groups[1], price_scaler, 'kmeans_1')
    start_regressor_expeirments(kmeans_groups[2], price_scaler, 'kmeans_2')

    svc_groups = create_groups(cluster_svm, K, x_train, y_train, x_test, y_test)
    start_regressor_expeirments(svc_groups[0], price_scaler, 'svc_0')
    start_regressor_expeirments(svc_groups[1], price_scaler, 'svc_1')
    start_regressor_expeirments(svc_groups[2], price_scaler, 'svc_2')
