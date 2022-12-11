from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from utils import LOCAL_DATA_DIR, default_args


with DAG(
    'download',
    default_args=default_args,
    schedule_interval='@daily',
) as dag:
    download = DockerOperator(
        image='airflow-download',
        command='--output-dir /data/raw/{{ ds }} --start-date {{ var.value.stock_start_date }} --end-date {{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-download',
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=LOCAL_DATA_DIR, target='/data', type='bind')]
    )

    download
