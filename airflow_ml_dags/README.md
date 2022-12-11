# Homework #03: airflow-pipeline
**Pipeline consists of three parts**:
1. *Every day* download Apple inc. stocks data from [yahoo finance](https://finance.yahoo.com/) from `start-date` to current date.
2. *Every week* preprocess data/extract features, split to train-val, fit catboost regressor and validate with saving metrics
3. *Every day* taking downloaded at same day data and last trained model(or other model which path will be in `path_to_model` airflow variable) and making predictions
### Getting started
1. Fill the .env file with your parameters or set them via terminal:
```commandline
export LOCAL_DATA_DIR=<path-to-data-dir> # Local directory where all data will be stored
export RANDOM_SEED=<number> # Random seed
export STOCK_START_DATE=<YYYY-MM-DD> # For example 2017-01-01
export USER_EMAIL=<email> # Email for getting messages when something goes wrong
export PASSWORD=<password> # Password for email
export PATH_TO_MODEL=<dummy/path/to/nowhere> # Needed as a bugfix
export FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
```
2. Run in the terminal
```commandline
sudo docker compose up --build
```