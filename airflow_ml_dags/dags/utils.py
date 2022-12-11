import os
from datetime import timedelta, datetime
from airflow.models import Variable
from airflow.utils.email import send_email_smtp


LOCAL_DATA_DIR = Variable.get('local_data_dir', default_var="./data")
PATH_TO_MODEL = Variable.get('path_to_model', default_var="/data/models/{{ ds }}")
RANDOM_SEED = Variable.get('random_seed', default_var=1488)
STOCK_START_DATE = Variable.get('stock_start_date', default_var="2017-01-01")


def wait_for_file(*filenames):
    return all(os.path.exists(file_name) for file_name in filenames)

def wait_for_models():
    model_files = [f"/opt/airflow{Variable.get('path_to_model', default_var=' ')}/standard_scaler_data.pkl",
                   f"/opt/airflow{Variable.get('path_to_model', default_var=' ')}/standard_scaler_target.pkl",
                   f"/opt/airflow{Variable.get('path_to_model', default_var=' ')}/catboost_regressor.cbm"]
    return wait_for_file(*model_files)

def failure_callback(context):
    dag_run = context.get('dag_run')
    subject = f'DAG {dag_run} has failed'
    send_email_smtp(to=default_args['email'], subject=subject)


default_args = {
    'owner': 'conor_mcgregor',
    'email': ['conor.mcgregor@gmail.com'],
    'start_date': datetime(2022, 11, 25),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': failure_callback
}
