import os
import click
import pandas as pd
import click
import yfinance as yf


@click.command("download")
@click.option('--output-dir')
@click.option('--start-date')
@click.option('--end-date')
def download(output_dir: str, start_date: str, end_date: str):
    os.makedirs(output_dir, exist_ok=True)
    df = yf.download(tickers="AAPL", start=start_date, end=end_date).reset_index(drop=True)
    df.to_csv(os.path.join(output_dir, "data.csv"), index=False)


if __name__ == '__main__':
    download()