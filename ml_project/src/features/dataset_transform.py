import pickle
from collections.abc import Iterable
import pandas as pd
from .normalize_and_encode_transformer import Normalizer


def dataset_fit_transform(X_train: pd.DataFrame,
                          cat_features: list[str],
                          target_features: list[str],
                          transformer_save_path: str,
                          Xs: Iterable[pd.DataFrame] = None) -> tuple:
    Xs = list() if Xs is None else Xs
    num_features = [colname for colname in X_train.columns.tolist()
                    if colname not in target_features + cat_features]
    transformer = Normalizer(num_features=num_features,
                             cat_features=cat_features)
    transformer.fit(X_train)
    with open(transformer_save_path, "wb") as f:
        pickle.dump(transformer, f)
    X_train = transformer.transform(X_train)
    transformed_Xs = list()
    for X in Xs:
        transformed_Xs.append(transformer.transform(X))
    return X_train, transformed_Xs


def dataset_transform(transformer_save_path: str,
                      *Xs) -> list:
    with open(transformer_save_path, "rb") as f:
        transformer = pickle.load(f)
    transformed_Xs = list()
    for X in Xs:
        transformed_Xs.append(transformer.transform(X))
    return transformed_Xs
