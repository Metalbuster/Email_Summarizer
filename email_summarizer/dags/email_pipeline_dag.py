from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="email_ml_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="cd /opt/airflow/project && python scripts/prepare_data.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow/project && python scripts/train_model.py",
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command="cd /opt/airflow/project && python scripts/evaluate_model.py",
    )

    prepare_data >> train_model >> evaluate_model
