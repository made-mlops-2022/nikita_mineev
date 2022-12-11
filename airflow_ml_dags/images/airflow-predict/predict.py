import os
import pickle
import click
import pandas as pd
from sklearn.preprocessing import StandardScaler
import catboost
from catboost import CatBoostRegressor


@click.command('predict')
@click.option('--data-dir')
@click.option('--models-dir')
@click.option('--predictions-dir')
def predict(data_dir: str, models_dir: str, predictions_dir: str):
    os.makedirs(predictions_dir, exist_ok=True)
    X = pd.read_csv(os.path.join(data_dir, 'data.csv'))

    with open(os.path.join(models_dir, "standard_scaler_data.pkl"), "rb") as f:
        ss_data = pickle.load(f)

    with open(os.path.join(models_dir, "standard_scaler_target.pkl"), "rb") as f:
        ss_target = pickle.load(f)

    regr = CatBoostRegressor()
    regr.load_model(os.path.join(models_dir, "catboost_regressor.cbm"))

    # Predicting
    X = ss_data.transform(X)
    y_pred = regr.predict(X)
    y_pred = ss_target.inverse_transform(y_pred.reshape(-1, 1))
    pd.DataFrame(y_pred, columns=["prediction"]).to_csv(
        os.path.join(predictions_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    predict()
