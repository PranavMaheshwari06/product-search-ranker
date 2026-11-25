"""Daily retrain: fetch -> prepare -> index -> train -> eval (candidate) -> gate -> publish.

Each task runs the `psr` CLI from its own venv in the image, so Airflow's pinned constraints never
meet the project's dependencies. PSR_DATA_MODE=fixtures (Compose default) runs it on the fixture.
Real mode expects the e5 embeddings already built (`psr embed`, a GPU job), as dense.py says.
"""

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

PSR = "/opt/psr-venv/bin/psr"
STEPS = {
    "fetch": "fetch --if-changed",
    "prepare": "prepare",
    "index": "index",
    "train": "train",
    "eval_candidate": "eval --systems ltr --candidate",
    "gate": "gate",
    "publish": "publish",
}

with DAG(
    dag_id="psr_daily",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["psr"],
) as dag:
    prev = None
    for task_id, args in STEPS.items():
        task = BashOperator(task_id=task_id, bash_command=f"{PSR} {args}")
        if prev is not None:
            prev >> task
        prev = task
