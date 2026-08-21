# model_building_step.py — English comments, model_type parametrized

import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, Model, step
from zenml.client import Client
from zenml import Model


mlflow.set_tracking_uri("http://localhost:5001")

# Get the active experiment tracker from ZenML
experimennt_tracker = Client().active_stack.experiment_tracker

model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)

# Registry mapping a simple string key to an actual sklearn estimator instance.
# This is the single place you touch to add a new candidate model.
MODEL_REGISTRY = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
}


@step(enable_cache=False, experiment_tracker=experimennt_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "linear_regression",  # NEW: controls which estimator gets trained
) -> Annotated[Pipeline, ArtifactConfig(name="skelearn_pipeline", is_model_artifact=True)]:
    """
    Builds and trains a regression model chosen by `model_type`, wrapped in a
    preprocessing pipeline (imputation + one-hot encoding).

    Parameters:
    X_train (pd.DataFrame): Training data features.
    y_train (pd.Series): Training data labels/target.
    model_type (str): Key into MODEL_REGISTRY selecting which estimator to train.

    Returns:
    Pipeline: The trained scikit-learn pipeline including preprocessing and the selected model.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas Dataframe")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series")

    # Fail fast with a clear error if someone passes an unsupported model_type
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose one of: {list(MODEL_REGISTRY.keys())}"
        )

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    # Define preprocessing for categorical and numerical features
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Pull the selected estimator from the registry instead of hardcoding LinearRegression
    selected_model = MODEL_REGISTRY[model_type]

    # Define the model training pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", selected_model)])

    # Start an MLflow run to log the model training process
    if not mlflow.active_run():
        # run_name shows the model type directly in the MLflow runs table
        mlflow.start_run(run_name=model_type)
    try:
        # Enable autologging for scikit-learn to automatically capture model metrics, parameters, and artifacts
        mlflow.sklearn.autolog()

        # Manual tag: lets you filter/group runs by model_type in the MLflow UI,
        # independent of whatever autolog captures on its own.
        mlflow.set_tag("model_type", model_type)

        logging.info(f"Building and training the {model_type} model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        # Log the columns that the model expects
        onehot_encoder = (
            pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        )
        onehot_encoder.fit(X_train[categorical_cols])
        expected_columns = numerical_cols.tolist() + list(
            onehot_encoder.get_feature_names_out(categorical_cols)
        )
        logging.info(f"Model expects the following columns: {expected_columns}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        # End the MLflow run
        mlflow.end_run()

    return pipeline