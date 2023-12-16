from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import optuna
from joblib import dump, load
from cluster_classifier import load_groups

class TrainGroupModels:
    
    def __init__(self, price_scaler: StandardScaler, data_groups: list[dict[str, np.ndarray]]):
        self.price_scaler = price_scaler
        self.data_groups = data_groups

    # # kmeans
    # def group_0(self) -> StackingRegressor:
    #     return StackingRegressor([
    #         ("LinearRegression", LinearRegression(fit_intercept=False, positive=True)),
    #         ("Ridge", Ridge(alpha=41.75, solver='auto')),
    #         ("MLPR", MLPRegressor(
    #                 hidden_layer_sizes=(30,75),
    #                 activation="identity",
    #                 solver="sgd",
    #                 alpha=0.0016829719012286812,
    #                 learning_rate="constant",
    #                 learning_rate_init=0.0036793704523117175,
    #                 max_iter=500)),
    #         ('Boosting', GradientBoostingRegressor(
    #             loss='squared_error',
    #             learning_rate=0.0524377156399538,
    #             n_estimators=510,
    #             subsample=0.1,
    #             alpha=0.75,
    #             warm_start=False))
    #     ])
    # def group_1(self) -> StackingRegressor:
    #     lr_params = {'fit_intercept': False, 'positive': True}
    #     svr_params = {"C": 7.013458510866372, "gamma": 'scale', "kernel": "linear"}
    #     mlpreg_params = {
    #         "activation": "logistic",
    #         "alpha": 7.425680314460657e-05,
    #         "hidden_layer_sizes": (55,30),
    #         "max_iter": 1_000_000,
    #         "solver": "sgd",
    #         "learning_rate": 'constant',
    #         'learning_rate_init': 0.0974030632736834,
    #     }
    #     gradientboost_params = {
    #         "learning_rate": 0.020722374937165538,
    #         # "max_depth": 7,
    #         "n_estimators": 560,
    #         "subsample": 0.2,
    #         "loss": "huber",
    #         "alpha": 0.35
    #     }
    #     estimators = [
    #         ("SVR", SVR(**svr_params)),
    #         ("Ridge", LinearRegression(**lr_params)),
    #         ("MLPRegressor", MLPRegressor(**mlpreg_params)),
    #         (
    #             "GradientBoostingRegressor",
    #             GradientBoostingRegressor(**gradientboost_params),
    #         ),
    #     ]
    #     return StackingRegressor(estimators)
    # def group_2(self) -> StackingRegressor:
    #     return StackingRegressor([
    #         ("LR", LinearRegression(fit_intercept=False, positive=False)),
    #         ("Ridge", Ridge(alpha=0.12, solver="sag")),
    #         ("MLPR", MLPRegressor(
    #             hidden_layer_sizes=(25,30),
    #             activation="logistic",
    #             solver="adam",
    #             alpha=0.003621301816841043,
    #             learning_rate="adaptive",
    #             learning_rate_init=0.002813955173521995,
    #             max_iter=400)),
    #         ('Boosting', GradientBoostingRegressor(
    #             loss='huber',
    #             learning_rate=0.09974297230400245,
    #             n_estimators=820,
    #             subsample=0.3,
    #             alpha=0.55,
    #             warm_start=True))     
    #     ])

    # svc
    def group_0(self) -> StackingRegressor:
        return StackingRegressor([
            ("Ridge", Ridge(alpha=98.2, solver="sag")),
            ("SVR", SVR(kernel="linear", gamma="scale", C=0.004728821687360534)),
            ("MLPR", MLPRegressor(
                hidden_layer_sizes=(10),
                activation="identity",
                solver="sgd",
                alpha=0.00020117678249370564,
                learning_rate="adaptive",
                learning_rate_init=0.023634736910184237,
                max_iter=400)),
            ('Boosting', GradientBoostingRegressor(
                loss='squared_error',
                learning_rate=0.0059274208314858585,
                n_estimators=500,
                subsample=0.6,
                alpha=0.55,
                warm_start=True))     
        ])
    def group_1(self) -> StackingRegressor:
        return StackingRegressor([
            ("Ridge", Ridge(alpha=139.53, solver="saga")),
            ("SVR", SVR(kernel="linear", gamma="scale", C=0.0764798603589879)),
            ("MLPR", MLPRegressor(
                hidden_layer_sizes=(60,55,60),
                activation="logistic",
                solver="adam",
                alpha=0.01679307587340253,
                learning_rate="invscaling",
                learning_rate_init=0.003879274041149979,
                max_iter=400)),
            ('Boosting', GradientBoostingRegressor(
                loss='absolute_error',
                learning_rate=0.04610825567818563,
                n_estimators=450,
                subsample=0.6,
                alpha=0.6,
                warm_start=True))     
        ])
    def group_2(self) -> StackingRegressor:
        return StackingRegressor([
            ("LR", LinearRegression(fit_intercept=True, positive=False)),
            ("Ridge", Ridge(alpha=0.02, solver="sparse_cg")),
            ("SVR", SVR(kernel="linear", gamma="scale", C=4.764139172447022)),
            ("MLPR", MLPRegressor(
                hidden_layer_sizes=(60,15),
                activation="logistic",
                solver="adam",
                alpha=0.0005788100531751306,
                learning_rate="invscaling",
                learning_rate_init=0.0034348144955733263,
                max_iter=400))   
        ])


    def group_model(self, group_id) -> StackingRegressor:
        models = {0: self.group_0, 1: self.group_1, 2: self.group_2}
        return models[group_id]

    # def load_data(self, group_id):
    #     return load_groups(self.group_ds_path(group_id))

    def train_and_evaluate(self, group_id: int):
        print(f"Training Group {group_id}")
        data = self.data_groups[group_id]
        stacked_model = self.group_model(group_id)()

        x_train = np.row_stack([data["x_train"], data["x_val"]])
        y_train = np.concatenate([data["y_train"], data["y_val"]])

        stacked_model.fit(x_train, y_train.flatten())

        train_prices = (
            self.price_scaler.inverse_transform(y_train.reshape(-1, 1)),
            self.price_scaler.inverse_transform(
                stacked_model.predict(x_train).reshape(-1, 1)
            ),
        )
        test_prices = (
            self.price_scaler.inverse_transform(data["y_test"].reshape(-1, 1)),
            self.price_scaler.inverse_transform(
                stacked_model.predict(data["x_test"]).reshape(-1, 1)
            ),
        )

        metrics = {
            "train_mse": mean_squared_error(*train_prices),
            "test_mse": mean_squared_error(*test_prices),
            "train_rmse": mean_squared_error(*train_prices, squared=False),
            "test_rmse": mean_squared_error(*test_prices, squared=False),
            "train_mae": mean_absolute_error(*train_prices),
            "test_mae": mean_absolute_error(*test_prices),
        }
        print(metrics)

        return stacked_model

""" Hyper tuning studies """

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
