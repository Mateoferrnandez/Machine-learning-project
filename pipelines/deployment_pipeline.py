# deployment_pipeline.py — English comments, model_type parametrized

import os

from pipelines.training_pipeline import ml_pipeline
from steps.dynamic_importer import dynamic_importer
from steps.model_loader import model_loader
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

import mlflow.pyfunc
requeriments_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@pipeline
def continuous_deployment_pipeline(model_type: str = "linear_regression"):
    """
    Retrains the chosen model_type end-to-end and deploys the result.

    Parameters:
    model_type (str): Which model to train and deploy. Comes from the CLI,
    not hardcoded, so deployment always reflects a deliberate choice made
    after comparing runs in MLflow — not whatever the pipeline default is.
    """
    # Explicit model_type instead of relying on ml_pipeline's own default
    trained_model = ml_pipeline(model_type=model_type)

    mlflow_model_deployer_step(workers=1, deploy_decision=True, model=trained_model, mlserver=True)


@pipeline(enable_cache=False)
def inference_pipeline():
    """Run a batch inference job with data loaded from an API."""
    batch_data = dynamic_importer()

    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )

    predictor(service=model_deployment_service, input_data=batch_data)