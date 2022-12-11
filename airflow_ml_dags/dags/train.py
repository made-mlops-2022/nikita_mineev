from datetime import datetime
from airflow.decorators import task
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from docker.types import Mount
from airflow.models import Variable
from utils import LOCAL_DATA_DIR, default_args, wait_for_file


def set_path_to_model(ds):
    Variable.set("path_to_model", f"/data/models/{ds}")


with DAG(
    'train',
    default_args=default_args,
    schedule_interval='@weekly',
) as dag:
    wait_data = PythonSensor(
        task_id='wait-for-data',
        python_callable=wait_for_file,
        op_args=['/opt/airflow/data/raw/{{ ds }}/data.csv'],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    preprocess = DockerOperator(
        image='airflow-preprocess',
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        network_mode='bridge',
        task_id='docker-airflow-preprocess',
        do_xcom_push=False,
        mounts=[Mount(source=LOCAL_DATA_DIR, target='/data', type='bind')]
    )

    split = DockerOperator(
        image='airflow-split',
        command='--input-dir /data/processed/{{ ds }} --output-dir /data/splitted/{{ ds }} --random-seed {{ var.value.random_seed }}',
        network_mode='bridge',
        task_id='docker-airflow-split',
        do_xcom_push=False,
        mounts=[Mount(source=LOCAL_DATA_DIR, target='/data', type='bind')]
    )

    train = DockerOperator(
        image='airflow-train',
        command='--data-dir /data/splitted/{{ ds }} --models-dir /data/models/{{ ds }} --random-seed {{ var.value.random_seed }}',
        network_mode='host',
        task_id='docker-airflow-train',
        do_xcom_push=True,
        mounts=[Mount(source=LOCAL_DATA_DIR, target='/data', type='bind')],
    )

    actualize_path_to_model = PythonOperator(task_id='actualize_path_to_model',
                                             python_callable=set_path_to_model)


    validate = DockerOperator(
        image='airflow-validate',
        command='--data-dir /data/splitted/{{ ds }} --models-dir /data/models/{{ ds }} --metrics-dir /data/metrics/{{ ds }}',
        network_mode='host',
        task_id='docker-airflow-validate',
        do_xcom_push=False,
        mounts=[Mount(source=LOCAL_DATA_DIR, target='/data', type='bind')]
    )

    wait_data >> preprocess >> split >> train >> actualize_path_to_model >> validate
