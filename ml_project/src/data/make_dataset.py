import pandas as pd
from sklearn.model_selection import train_test_split


def make_dataset(datapath: str,
                 train_size: int,
                 val_size: int,
                 test_size: int,
                 random_state: int) -> list:
    dataset = [pd.read_csv(datapath)]
    if train_size and (test_size or val_size):
        X_train, X_test = train_test_split(*dataset, test_size=test_size,
                                           random_state=random_state)
        X_train, X_val = train_test_split(*dataset, test_size=val_size,
                                          random_state=random_state)
        dataset = [X_train, X_val, X_test]

    return dataset
