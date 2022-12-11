import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command('split')
@click.option('--input-dir')
@click.option('--output-dir')
@click.option('--random-seed')
def split(input_dir: str, output_dir: str, random_seed: int):
    os.makedirs(output_dir, exist_ok=True)
    random_seed = int(random_seed)
    X = pd.read_csv(os.path.join(input_dir, 'data.csv'))
    y = pd.read_csv(os.path.join(input_dir, 'target.csv'))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    X_train.to_csv(os.path.join(output_dir, 'data_train.csv'), index=False)
    X_val.to_csv(os.path.join(output_dir, 'data_val.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'target_train.csv'), index=False)
    y_val.to_csv(os.path.join(output_dir, 'target_val.csv'), index=False)


if __name__ == '__main__':
    split()
