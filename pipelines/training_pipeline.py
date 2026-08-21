# training_pipeline.py — English comments, model_type passed through

from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step
from zenml import Model, pipeline


@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="sells_pridector"
    ),
)
def ml_pipeline(model_type: str = "linear_regression"):
    """
    Define an end-to-end machine learning pipeline.

    Parameters:
    model_type (str): Which model to train this run. Passed straight through
    to model_building_step, so calling ml_pipeline(model_type="random_forest")
    creates a fully independent, comparable run without editing this file.
    """
    # Data Ingestion Step
    raw_data = data_ingestion_step(
        "/Users/mateofernandez/Documents/Github/Machine-learning-project/data/Datosmodelo.csv"
    )

    # Handling Missing Values Step
    filled_data = handle_missing_values_step(raw_data)

    # Feature Engineering Step
    engineered_data = feature_engineering_step(
        filled_data, strategy="onehot_encoding", features=['TALLA', 'NOMB_SUBGRUPO', 'CAMPANA']
    )

    # Outlier Detection Step
    clean_data = outlier_detection_step(engineered_data)

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="VENTA")

    # Model Building Step — model_type flows from the pipeline parameter
    model = model_building_step(X_train=X_train, y_train=y_train, model_type=model_type)

    # Model Evaluation Step — unchanged, works generically for any sklearn model
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model


if __name__ == "__main__":
    # Running the pipeline with the default model_type
    run = ml_pipeline()