import json

import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (str): The input data as a JSON string.

    Returns:
        np.ndarray: The model's prediction.
    """

    # Start the service (should be a NOP if already started)
    service.start(timeout=10)

    # Load the input data from JSON string
    data = json.loads(input_data)

    # Extract the actual data and expected columns
    data.pop("columns", None)  # Remove 'columns' if it's present
    data.pop("index", None)  # Remove 'index' if it's present

    # Define the columns the model expects
    expected_columns = ['VENTA_ZONA_101', 'VENTA_ZONA_102', 'VENTA_ZONA_103', 'VENTA_ZONA_104', 'VENTA_ZONA_107', 'VENTA_ZONA_109', 'VENTA_ZONA_110', 'VENTA_ZONA_111', 'VENTA_ZONA_112', 'VENTA_ZONA_115', 'VENTA_ZONA_116', 'VENTA_ZONA_119', 'PRECIO_NAC', 'N° ASESORAS_ZONA_101', 'N° ASESORAS_ZONA_102', 'N° ASESORAS_ZONA_103', 'N° ASESORAS_ZONA_104', 'N° ASESORAS_ZONA_107', 'N° ASESORAS_ZONA_109', 'N° ASESORAS_ZONA_110', 'N° ASESORAS_ZONA_111', 'N° ASESORAS_ZONA_112', 'N° ASESORAS_ZONA_115', 'N° ASESORAS_ZONA_116', 'N° ASESORAS_ZONA_119', 'N° ASESORAS', 'TALLA_T-12', 'TALLA_T-14', 'TALLA_T-16', 'TALLA_T-6', 'TALLA_T-8', 'TALLA_T-L', 'TALLA_T-M', 'TALLA_T-S', 'TALLA_T-UNI', 'TALLA_T-XL', 'TALLA_T-XS', 'TALLA_T-XXL', 'NOMB_SUBGRUPO_201 RE-BLUSAS FEM', 'NOMB_SUBGRUPO_202 RE-BODYS FEM', 'NOMB_SUBGRUPO_204 RE-BUZOS FEM', 'NOMB_SUBGRUPO_206 RE-CAMISAS FEM', 'NOMB_SUBGRUPO_207 RE-CAMISETAS FEM', 'NOMB_SUBGRUPO_209 RE-CAPRIS FEM', 'NOMB_SUBGRUPO_210 RE-CHALECOS FEM', 'NOMB_SUBGRUPO_211 RE-CHAQUETAS FEM', 'NOMB_SUBGRUPO_213 RE-CONJUNTOS FEM', 'NOMB_SUBGRUPO_214 RE-ENTERIZOS FEM', 'NOMB_SUBGRUPO_215 RE-FALDAS FEM', 'NOMB_SUBGRUPO_216 RE-JEANS FEM', 'NOMB_SUBGRUPO_217 RE-JOGGERS FEM', 'NOMB_SUBGRUPO_218 RE-LEGGINS FEM', 'NOMB_SUBGRUPO_219 RE-OVEROLES FEM', 'NOMB_SUBGRUPO_220 RE-PANTALONES FEM', 'NOMB_SUBGRUPO_221 RE-PESCADORES FEM', 'NOMB_SUBGRUPO_223 RE-SHORTS FEM', 'NOMB_SUBGRUPO_224 RE-SOBRETODOS FEM', 'NOMB_SUBGRUPO_225 RE-VESTIDOS FEM', 'CAMPANA_201902', 'CAMPANA_201903', 'CAMPANA_201904', 'CAMPANA_201905', 'CAMPANA_201906', 'CAMPANA_201907', 'CAMPANA_201908', 'CAMPANA_201909', 'CAMPANA_201910', 'CAMPANA_201911', 'CAMPANA_201912']

    # Convert the data into a DataFrame with the correct columns
    df = pd.DataFrame(data["data"], columns=expected_columns)

    # Convert DataFrame to JSON list for prediction
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    # Run the prediction
    prediction = service.predict(data_array)

    return prediction