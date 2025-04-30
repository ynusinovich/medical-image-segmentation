# scripts/utils/mlflow_logger.py

import mlflow


class MLFlowLogger:
    def __init__(self, experiment_name, run_name):
        self.experiment_name = experiment_name
        self.run_name = run_name

        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, filepath):
        mlflow.log_artifact(filepath)

    def end_run(self):
        mlflow.end_run()
