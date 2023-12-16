import numpy as np

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

class TrainGroupModels:
    
    def __init__(self, price_scaler: StandardScaler, data_groups: list[dict[str, np.ndarray]]):
        self.price_scaler = price_scaler
        self.data_groups = data_groups

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
