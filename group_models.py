from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import optuna
from cluster_classifier import load_groups

"""
NOTE: the data dictionaries have the following keys:
    x_train
    y_train
    x_val
    y_val
    x_test
    y_test
"""


class TrainGroupModels:
    def __init__(self, price_scaler: StandardScaler):
        self.price_scaler = price_scaler

    def group_0(self) -> StackingRegressor:
        return StackingRegressor(
            [
                ("LinearRegression", LinearRegression()),
                ("Ridge", Ridge(alpha=10, solver="saga")),
                ("SVR", SVR(kernel="linear", gamma="scale", C=2.3609999999999998)),
                (
                    "MLPR",
                    MLPRegressor(
                        hidden_layer_sizes=(30, 20),
                        activation="identity",
                        solver="adam",
                        alpha=0.005559340227398926,
                        learning_rate="adaptive",
                        learning_rate_init=0.0006506360597208436,
                    ),
                ),
            ]
        )

    def group_1(self) -> StackingRegressor:
        svr_params = {"C": 1, "gamma": 0.1, "kernel": "linear"}
        ridge_paras = {"alpha": 100}
        mlpreg_params = {
            "activation": "logistic",
            "alpha": 5e-05,
            "hidden_layer_sizes": (50,),
            "max_iter": 1_000_000,
            "solver": "sgd",
        }
        gradientboost_params = {
            "learning_rate": 0.01,
            "max_depth": 7,
            "n_estimators": 500,
            "subsample": 0.5,
        }
        estimators = [
            ("SVR", SVR(**svr_params)),
            ("Ridge", Ridge(**ridge_paras)),
            ("MLPRegressor", MLPRegressor(**mlpreg_params)),
            (
                "GradientBoostingRegressor",
                GradientBoostingRegressor(**gradientboost_params),
            ),
        ]
        return StackingRegressor(estimators)

    def group_2(self) -> StackingRegressor:
        pass

    def group_ds_path(self, group_id):
        return f"datasets/groups/{group_id}"

    def group_model(self, group_id) -> StackingRegressor:
        models = {0: self.group_0, 1: self.group_1, 2: self.group_1}
        return models[group_id]

    def load_data(self, group_id):
        return load_groups(self.group_ds_path(group_id))

    def train_and_evaluate(self, group_id: str):
        data = self.load_data(group_id)
        stacked_model = self.group_model(group_id)()

        x_train = np.row_stack([data["x_train"], data["x_val"]])
        y_train = np.concatenate([data["y_train"], data["y_val"]])

        stacked_model.fit(x_train, y_train)

        train_mse = mean_squared_error(
            self.price_scaler.inverse_transform(y_train.reshape(-1, 1)),
            self.price_scaler.inverse_transform(
                stacked_model.predict(x_train).reshape(-1, 1)
            ),
        )

        print(f"Train MSE is {train_mse}")

        test_mse = mean_squared_error(
            self.price_scaler.inverse_transform(data["y_test"].reshape(-1, 1)),
            self.price_scaler.inverse_transform(
                stacked_model.predict(data["x_test"]).reshape(-1, 1)
            ),
        )

        print(f"Test MSE is {test_mse}")
        return stacked_model, train_mse, test_mse


""" Hyper tuning studies """


def tune_lr(data: dict[str, np.ndarray], scaler, db_name="db"):
    def regressor_trial(trial: optuna.Trial):
        regressor = LinearRegression(
            fit_intercept=trial.suggest_categorical("lr_fit_intercept", [True, False]),
            positive=trial.suggest_categorical("lr_positive", [True, False]),
        )

        regressor.fit(data["x_train"], data["y_train"])
        return mean_squared_error(
            scaler.inverse_transform(data["y_val"].reshape(-1, 1)),
            scaler.inverse_transform(regressor.predict(data["x_val"]).reshape(-1, 1)),
        )

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
        return mean_squared_error(
            scaler.inverse_transform(data["y_val"].reshape(-1, 1)),
            scaler.inverse_transform(regressor.predict(data["x_val"]).reshape(-1, 1)),
        )

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
        return mean_squared_error(
            scaler.inverse_transform(data["y_val"].reshape(-1, 1)),
            scaler.inverse_transform(regressor.predict(data["x_val"]).reshape(-1, 1)),
        )

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

        regressor.fit(data["x_train"], data["y_train"])
        return mean_squared_error(
            scaler.inverse_transform(data["y_val"].reshape(-1, 1)),
            scaler.inverse_transform(regressor.predict(data["x_val"]).reshape(-1, 1)),
        )

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
        return mean_squared_error(
            scaler.inverse_transform(data["y_val"].reshape(-1, 1)),
            scaler.inverse_transform(regressor.predict(data["x_val"]).reshape(-1, 1)),
        )

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
            layers.append(trial.suggest_int(f"mlpr_{layer}_neurons", 5, 100, step=5))

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
            regressor.fit(data["x_train"], data["y_train"])
            return mean_squared_error(
                scaler.inverse_transform(data["y_val"].reshape(-1, 1)),
                scaler.inverse_transform(
                    regressor.predict(data["x_val"]).reshape(-1, 1)
                ),
            )
        except:
            # an exception is thrown if the weights or predictions become infinite
            return None

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{db_name}.sqlite3",
        study_name="mlpr",
        load_if_exists=True,
    )

    study.optimize(regressor_trial, n_trials=500)


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

    scores = []
    for study in optuna.study.get_all_study_summaries(
        storage=f"sqlite:///{db_name}.sqlite3"
    ):
        print(study.best_trial.params)
        print(study.best_trial.value, end="\n\n")
        scores.append(study.best_trial.value)
    print(f"best index = {np.array(scores).argmin()}")
