import os
import click
import pandas as pd


@click.command('preprocess')
@click.option('--input-dir', type=click.Path())
@click.option('--output-dir', type=click.Path())
def preprocess(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(input_dir, "data.csv"))
    df = pd.concat([df, *[df.shift(i).add_suffix(f"_lag{i}") for i in range(1, 5)]], axis=1).dropna()
    target = df["Close"].copy()
    df.drop(columns=["Open", "High", "Low", "Adj Close", "Volume", "Close"], inplace=True)
    df.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == '__main__':
    preprocess()
