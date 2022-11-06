from typing import Union
import pandas as pd
import numpy as np
import catboost
from catboost import CatBoostClassifier
import time
import os
from sklearn.metrics import f1_score
from features.dataset_transform import dataset_transform


def predict_model(X_test: pd.DataFrame,
                  cat_features: list[str],
                  model_save_path: str,
                  save_predict_path: str,
                  transform_dataset: bool = False,
                  transformer_save_path: str = None) -> None:
    if transform_dataset:
        X_test, = dataset_transform(transformer_save_path, X_test)
    clf = CatBoostClassifier()
    clf.load_model(model_save_path)
    test_pool = catboost.Pool(data=X_test, cat_features=cat_features)
    pred = clf.predict(test_pool)
    np.savetxt(save_predict_path, pred, delimiter=",")
