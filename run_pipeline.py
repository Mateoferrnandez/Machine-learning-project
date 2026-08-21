# run_pipeline.py — English comments, loop over multiple model types

import click
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


@click.command()
@click.option(
    "--model-type",          # Flag tal como lo escribe el usuario en la terminal
    "model_types",             # Nombre de la variable que click inyecta en main() (plural porque será una tupla)
    multiple=True,             # Permite repetir el flag varias veces; click acumula los valores en una tupla
                                # Ej: --model-type linear_regression --model-type random_forest
                                #     -> model_types = ("linear_regression", "random_forest")
    default=["linear_regression"],  # Valor por defecto si no se pasa el flag; debe ser una lista/colección
                                     # porque multiple=True siempre produce una tupla, no un escalar
    help=(
        "Model type(s) to train. Repeat the flag to train several, e.g. "
        "--model-type linear_regression --model-type random_forest"
    ),  # Texto mostrado al correr `python run_pipeline.py --help`
)
def main(model_types):
    """
    Run the ML pipeline once per model type so results land as separate,
    comparable MLflow runs.
    """
    for model_type in model_types:
        print(f"Training model_type={model_type} ...")
        # Each call creates a new ZenML pipeline run and a new MLflow run
        ml_pipeline(model_type=model_type)

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "Select the runs you want to compare and click 'Compare'."
    )


if __name__ == "__main__":
    main()