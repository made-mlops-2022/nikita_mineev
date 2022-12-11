import os
import pickle

import click
import pandas as pd
from sklearn.preprocessing import StandardScaler
import catboost
from catboost import CatBoostRegressor


@click.command('train')
@click.option('--data-dir')
@click.option('--models-dir')
@click.option('--random-seed')
def train(data_dir: str, models_dir: str, random_seed: int):
    os.makedirs(models_dir, exist_ok=True)
    random_seed = int(random_seed)
    # Loading
    X_train = pd.read_csv(os.path.join(data_dir, "data_train.csv"))
    X_val = pd.read_csv(os.path.join(data_dir, "data_val.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "target_train.csv"))
    y_val = pd.read_csv(os.path.join(data_dir, "target_val.csv"))

    # Preprocessing
    ss_data = StandardScaler()
    ss_target = StandardScaler()
    X_train = ss_data.fit_transform(X_train)
    X_val = ss_data.transform(X_val)
    y_train = ss_target.fit_transform(y_train)
    y_val = ss_target.transform(y_val)

    # Training
    train_pool = catboost.Pool(X_train, label=y_train)
    val_pool = catboost.Pool(X_val, label=y_val)

    regr = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        # l2_leaf_reg=1,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=random_seed,
        use_best_model=True,
        # early_stopping_rounds=20,
        metric_period=1,
        verbose=100,
        task_type="CPU"
    )

    regr.fit(train_pool, eval_set=val_pool)

    # Saving
    with open(os.path.join(models_dir, "standard_scaler_data.pkl"), "wb") as f:
        pickle.dump(ss_data, f)

    with open(os.path.join(models_dir, "standard_scaler_target.pkl"), "wb") as f:
        pickle.dump(ss_target, f)

    regr.save_model(os.path.join(models_dir, "catboost_regressor.cbm"))


if __name__ == '__main__':
    train()
