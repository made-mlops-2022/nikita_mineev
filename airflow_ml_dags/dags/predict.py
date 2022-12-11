from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from docker.types import Mount
from utils import LOCAL_DATA_DIR, default_args, wait_for_file, wait_for_models


with DAG(
    'predict',
    default_args=default_args,
    schedule_interval='@daily',
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

    wait_model = PythonSensor(
        task_id='wait-for-model',
        python_callable=wait_for_models,
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    preprocess = DockerOperator(
        image='airflow-preprocess',
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/predict_processed/{{ ds }}",
        network_mode='bridge',
        task_id='docker-airflow-predict_preprocess',
        do_xcom_push=False,
        mounts=[Mount(source=LOCAL_DATA_DIR, target='/data', type='bind')]
    )

    predict = DockerOperator(
        image='airflow-predict',
        command=('--data-dir /data/predict_processed/{{ ds }} '
                 '--models-dir {{ var.value.path_to_model }} '
                 '--predictions-dir /data/predictions/{{ ds }}'),
        network_mode='host',
        task_id='docker-airflow-predict',
        do_xcom_push=False,
        mounts=[Mount(source=LOCAL_DATA_DIR, target='/data', type='bind')]
    )

    [wait_data, wait_model] >> preprocess >> predict
