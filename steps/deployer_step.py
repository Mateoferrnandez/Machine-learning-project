from zenml import step, get_step_context
from zenml.client import Client
from zenml.integrations.mlflow.services import MLFlowDeploymentConfig
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step(enable_cache=False)
def deploy_model_step(trained_model_artifact) -> MLFlowDeploymentService:
    zenml_client = Client()
    model_deployer = zenml_client.active_stack.model_deployer
    model_obj = trained_model_artifact.read()  # o trained_model_artifact.get() depende del tipo
    model_uri = model_obj.model_uri  # accede al atributo model_uri del objeto


    mlflow_deployment_config = MLFlowDeploymentConfig(
        name="mlflow-model-deployment-windows",
        description="Despliegue con blocking=True para Windows",
        pipeline_name=get_step_context().pipeline_name,
        pipeline_step_name=get_step_context().step_name,
        model_uri=model_uri,
        model_name="model",
        workers=1,
        mlserver=False,
        blocking=True,  # <-- esto es clave para Windows
        timeout=300,
    )

    service = model_deployer.deploy_model(
        config=mlflow_deployment_config,
        service_type=MLFlowDeploymentService.SERVICE_TYPE,
    )

    return service
