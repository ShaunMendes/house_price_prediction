from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import optuna

'''
NOTE: the data dictionaries have the following keys:
    x_train
    y_train
    x_val
    y_val
    x_test
    y_test
'''

def group_0(data: dict[str, np.ndarray], scaler: StandardScaler):
    '''
    TODO: merge the training and validation data together now that hypertuning is complete?
    '''

    group_0_model = StackingRegressor([
        ('LinearRegression', LinearRegression()),
        ('Ridge', Ridge(alpha=10, solver='saga')),
        ('SVR', SVR(kernel='linear', gamma='scale', C=2.3609999999999998)),
        ('MLPR', MLPRegressor(
            hidden_layer_sizes=(30,20),
            activation='identity',
            solver='adam',
            alpha=0.005559340227398926,
            learning_rate='adaptive',
            learning_rate_init=0.0006506360597208436
        ))
    ])

    # now that tuning is complete, merge the training and validation data. 
    # i believe the regressor performs a k fold anyways
    x_train = np.row_stack([data['x_train'], data['x_val']])
    y_train = np.concatenate([data['y_train'], data['y_val']])

    group_0_model.fit(x_train, y_train)

    train_mse = mean_squared_error(
        scaler.inverse_transform(y_train.reshape(-1, 1)) , 
        scaler.inverse_transform(group_0_model.predict(x_train).reshape(-1, 1))
    )

    test_mse = mean_squared_error(
        scaler.inverse_transform(data['y_test'].reshape(-1, 1)) , 
        scaler.inverse_transform(group_0_model.predict(data['x_test']).reshape(-1, 1))
    )

    return group_0_model, train_mse, test_mse

def group_1(data: dict[str, np.ndarray]):
    '''
    TODO: Create and train a StackingRegressor to predict prices for this group
    '''
    pass

def group_2(data: dict[str, np.ndarray]):
    '''
    TODO: Create and train a StackingRegressor to predict prices for this group
    '''
    pass

def group_3(data: dict[str, np.ndarray]):
    '''
    TODO: Create and train a StackingRegressor to predict prices for this group
    '''
    pass


''' Hyper tuning studies '''

def tune_lr(data: dict[str, np.ndarray], scaler):

    def regressor_trial(trial: optuna.Trial):
        regressor = LinearRegression(
            fit_intercept = trial.suggest_categorical('lr_fit_intercept', [True, False]),
            positive = trial.suggest_categorical('lr_positive', [True, False])
        )

        regressor.fit(data['x_train'], data['y_train'])
        return mean_squared_error(
            scaler.inverse_transform(data['y_val'].reshape(-1, 1)), 
            scaler.inverse_transform(regressor.predict(data['x_val']).reshape(-1, 1))
        )

    study = optuna.create_study(
        direction='minimize', 
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="lr",
        load_if_exists=True
    )

    study.optimize(regressor_trial, n_trials=4)

def tune_lasso(data: dict[str, np.ndarray], scaler):
        
    def regressor_trial(trial: optuna.Trial):
        regressor = Lasso(
            selection = trial.suggest_categorical('lasso_selection', ['cyclic', 'random'])
        )

        regressor.fit(data['x_train'], data['y_train'])
        return mean_squared_error(
            scaler.inverse_transform(data['y_val'].reshape(-1, 1)) , 
            scaler.inverse_transform(regressor.predict(data['x_val']).reshape(-1, 1))
        )

    study = optuna.create_study(
        direction='minimize', 
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="lasso",
        load_if_exists=True
    )

    study.optimize(regressor_trial, n_trials=2)

def tune_ridge(data: dict[str, np.ndarray], scaler):
    
    def regressor_trial(trial: optuna.Trial):
        regressor = Ridge(
            alpha = trial.suggest_float('ridge_alpha', 0, 10, step=0.25),
            solver = trial.suggest_categorical('ridge_solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
        )

        regressor.fit(data['x_train'], data['y_train'])
        return mean_squared_error(
            scaler.inverse_transform(data['y_val'].reshape(-1, 1)) , 
            scaler.inverse_transform(regressor.predict(data['x_val']).reshape(-1, 1))
        )

    study = optuna.create_study(
        direction='minimize', 
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="ridge",
        load_if_exists=True
    )

    study.optimize(regressor_trial, n_trials=200)

def tune_svr(data: dict[str, np.ndarray], scaler):
    
    def regressor_trial(trial: optuna.Trial):

        regressor = SVR(
            kernel = trial.suggest_categorical('svr_kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            degree = trial.suggest_int('svr_degree', 1, 5),
            gamma = trial.suggest_categorical('svr_gamma', ['scale', 'auto']),
            C = trial.suggest_float('svr_C', 0.001, 3.991, step=0.01)
        )

        regressor.fit(data['x_train'], data['y_train'])
        return mean_squared_error(
            scaler.inverse_transform(data['y_val'].reshape(-1, 1)) , 
            scaler.inverse_transform(regressor.predict(data['x_val']).reshape(-1, 1))
        )


    study = optuna.create_study(
        direction='minimize', 
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="svr",
        load_if_exists=True
    )

    study.optimize(regressor_trial, n_trials=250)

def tune_decision_tree(data: dict[str, np.ndarray], scaler):
    
    def regressor_trial(trial: optuna.Trial):
        regressor = DecisionTreeRegressor(
            splitter = trial.suggest_categorical('dt_splitter', ['best', 'random'])
        )

        regressor.fit(data['x_train'], data['y_train'])
        return mean_squared_error(
            scaler.inverse_transform(data['y_val'].reshape(-1, 1)) , 
            scaler.inverse_transform(regressor.predict(data['x_val']).reshape(-1, 1))
        )

    study = optuna.create_study(
        direction='minimize', 
        storage="sqlite:///db.sqlite3",
        study_name="dt",
        load_if_exists=True
    )

    study.optimize(regressor_trial, n_trials=2)

def tune_mlrp(data: dict[str, np.ndarray], scaler):
    
    def regressor_trial(trial: optuna.Trial):
        
        layers = []
        for layer in range(trial.suggest_int('mlpr_num_layers', 1, 4)):
            layers.append(trial.suggest_int(f'mlpr_{layer}_neurons', 10, 100, step=10))

        regressor = MLPRegressor(
            hidden_layer_sizes=layers,
            activation = trial.suggest_categorical('mlpr_activation', ['relu', 'identity', 'logistic', 'tanh']),
            solver = trial.suggest_categorical('mlpr_solver', ['sgd', 'adam']),
            alpha = trial.suggest_float('mlrp_alpha', 1e-5, 1e-1, log=True),
            learning_rate = trial.suggest_categorical('mlrp_lr', ['constant', 'invscaling', 'adaptive']),
            learning_rate_init = trial.suggest_float('mlrp_lr_init', 1e-5, 1e-1, log=True)
        )

        try:
            regressor.fit(data['x_train'], data['y_train'])
        except:
            # an exception is thrown if the weights become infinite
            return None

        return mean_squared_error(
            scaler.inverse_transform(data['y_val'].reshape(-1, 1)) , 
            scaler.inverse_transform(regressor.predict(data['x_val']).reshape(-1, 1))
        )

    study = optuna.create_study(
        direction='minimize', 
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="mlpr",
        load_if_exists=True
    )

    study.optimize(regressor_trial, n_trials=500)

def start_regressor_expeirments(data, price_scaler):
    tune_lr(data, price_scaler)
    tune_lasso(data, price_scaler)
    tune_ridge(data, price_scaler)
    tune_svr(data, price_scaler)
    tune_decision_tree(data, price_scaler)
    tune_mlrp(data, price_scaler)