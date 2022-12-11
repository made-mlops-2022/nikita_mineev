import json
import os
import pickle

import click
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import catboost
from catboost import CatBoostRegressor


@click.command('validate')
@click.option('--data-dir')
@click.option('--models-dir')
@click.option('--metrics-dir')
def validate(data_dir: str, models_dir: str, metrics_dir: str):
    os.makedirs(metrics_dir, exist_ok=True)
    # Loading
    X_val = pd.read_csv(os.path.join(data_dir, "data_val.csv"))
    y_val = pd.read_csv(os.path.join(data_dir, "target_val.csv"))

    with open(os.path.join(models_dir, "standard_scaler_data.pkl"), "rb") as f:
        ss_data = pickle.load(f)

    with open(os.path.join(models_dir, "standard_scaler_target.pkl"), "rb") as f:
        ss_target = pickle.load(f)

    regr = CatBoostRegressor()
    regr.load_model(os.path.join(models_dir, "catboost_regressor.cbm"))

    # Predicting
    X_val = ss_data.transform(X_val)
    y_val_pred = regr.predict(X_val)
    y_val_pred = ss_target.inverse_transform(y_val_pred.reshape(-1, 1))

    # Saving metrics
    with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as f:
        json.dump({"rmse": mean_squared_error(y_val, y_val_pred, squared=False),
                   "mape": mean_absolute_percentage_error(y_val, y_val_pred)},
                  f)


if __name__ == '__main__':
    validate()
