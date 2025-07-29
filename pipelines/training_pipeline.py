from steps.data_ingestion_step import data_ingestion_step
#from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
#from steps.model_building_step import model_building_step
#from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step
from zenml import Model, pipeline, step

@pipeline(
    model=Model(
        #The name uniquely identifies this model
        name="sells_pridector"
        ),
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""
    #Data Ingestion Step
    raw_data = data_ingestion_step("D:\Documentos\Github\Machine-learning-end-to-end-project\Price-predictor system propio\data\Datosmodelo.csv")

    #   Handling Missing Values Step
    filled_data = handle_missing_values_step(raw_data)

    # Feature Engineering Step
    engineered_data = feature_engineering_step(filled_data,strategy="onehot_encoding",features=['CODIGO_DEL_PRODUCTO','VENTA', 'VENTA_ZONA_101',
       'VENTA_ZONA_102', 'VENTA_ZONA_103', 'VENTA_ZONA_104', 'VENTA_ZONA_107',
       'VENTA_ZONA_109', 'VENTA_ZONA_110', 'VENTA_ZONA_111', 'VENTA_ZONA_112',
       'VENTA_ZONA_115', 'VENTA_ZONA_116', 'VENTA_ZONA_119','TALLA','NOMB_SUBGRUPO','CAMPANA']
    )

    # Outlier Detection Step
    clean_data = outlier_detection_step(engineered_data)
                                        
if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
