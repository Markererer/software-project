from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient()

def get_mlflow_client():
    return client