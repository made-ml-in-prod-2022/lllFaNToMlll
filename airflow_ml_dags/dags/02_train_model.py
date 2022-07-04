from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    'owner': 'airflow',
    'email': ['airflow@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
        '02_train_model',
        default_args=default_args,
        description='DAG пайплайна для обучения модели',
        schedule_interval="@weekly",
        start_date=days_ago(2),
) as dag:
    files = DockerOperator(
        task_id='Download_data',
        image='airflow-preprocessing-files',
        network_mode='bridge',
        do_xcom_push=False,
        mounts=[
            Mount(
                source='C:/Users/miair/PycharmProjects/ML_in_prod_HW1/airflow_ml_dags/data',
                target='/data',
                type='bind',
            )],
        command='--in_path /data/raw/{{ ds }} --tmp_path /data/tmp/{{ ds }}',
    )

    preprocessing = DockerOperator(
        task_id='Preprocessing_dataset',
        image='airflow-preprocessing',
        network_mode='bridge',
        do_xcom_push=False,
        mounts=[
            Mount(
                source='C:/Users/miair/PycharmProjects/ML_in_prod_HW1/airflow_ml_dags/data',
                target='/data',
                type='bind',
            )],
        command='--scaler_path /data/models/{{ ds }} --tmp_path /data/tmp/{{ ds }} --preprocess_path /data/preprocessed/{{ ds }}',
    )

    split = DockerOperator(
        task_id='Train_test_split',
        image='airflow-split',
        network_mode="bridge",
        do_xcom_push=False,
        mounts=[
            Mount(
                source='C:/Users/miair/PycharmProjects/ML_in_prod_HW1/airflow_ml_dags/data',
                target='/data',
                type='bind',
            )],
        command='--preprocessed_path /data/preprocessed/{{ ds }} '
                '--splitted_path /data/splitted/{{ ds }} '
                '--test_size 0.15 '
                '--random_state 42',
    )

    train = DockerOperator(
        task_id='Train_model',
        image='airflow-train',
        network_mode="bridge",
        do_xcom_push=False,
        mounts=[
            Mount(
                source='C:/Users/miair/PycharmProjects/ML_in_prod_HW1/airflow_ml_dags/data',
                target='/data',
                type='bind',
            )],
        command='--load_data_path /data/splitted/{{ ds }} '
                '--save_model_path /data/models/{{ ds }}',
    )

    validation = DockerOperator(
        task_id='Validate_model',
        image='airflow-validation',
        network_mode='bridge',
        do_xcom_push=False,
        mounts=[
            Mount(
                source='C:/Users/miair/PycharmProjects/ML_in_prod_HW1/airflow_ml_dags/data',
                target='/data',
                type='bind',
            )],
        command='--validation_path /data/splitted/{{ ds }} '
                '--metrics_path /data/models/{{ ds }} '
                '--model_path /data/models/{{ ds }}',
    )

    files >> preprocessing >> split >> train >> validation
