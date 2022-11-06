import pandas as pd
import catboost
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from features.dataset_transform import dataset_fit_transform


def train_model(X_train: pd.DataFrame,
                X_val: pd.DataFrame,
                X_test: pd.DataFrame,
                cat_features: list[str],
                target_features: list[str],
                model_save_path: str,
                transform_dataset: bool = False,
                transformer_save_path: str = None,
                logger=None,
                **model_params) -> None:
    if transform_dataset:
        X_train, [X_val, X_test] = dataset_fit_transform(X_train,
                                                         cat_features,
                                                         target_features,
                                                         transformer_save_path,
                                                         [X_val, X_test])
    pred_features = [colname for colname in X_train.columns.tolist()
                     if colname not in target_features]
    train_pool = catboost.Pool(data=X_train[pred_features],
                               label=X_train[target_features],
                               cat_features=cat_features)
    val_pool = catboost.Pool(data=X_val[pred_features],
                             label=X_val[target_features],
                             cat_features=cat_features)
    test_pool = catboost.Pool(data=X_test[pred_features],
                              label=X_test[target_features],
                              cat_features=cat_features)
    clf = CatBoostClassifier(**model_params)
    clf.fit(train_pool, eval_set=val_pool, plot=False)
    clf.save_model(model_save_path)
    if X_test.shape[0]:
        score = f1_score(X_test[target_features], clf.predict(test_pool))
        logger.info(f"Test f1 score: {score:.3f}")
