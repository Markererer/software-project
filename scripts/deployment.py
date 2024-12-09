import time
import argparse
from mlflow_client import get_mlflow_client
import mlflow
import joblib
import os
from lr_wrapper import lr_wrapper
import xgboost as xgb

def save_model(loaded_model, output_dir):
    # Get the flavors metadata
    flavors = loaded_model.metadata.flavors

    # Check if the model is XGBoost
    if "xgboost" in flavors:
        print("Detected XGBoost model.")
        # Get the artifact path for the XGBoost model
        artifact_path = flavors["xgboost"]["data"]

        # Download the artifact to a local directory
        local_model_path = mlflow.artifacts.download_artifacts(artifact_path)

        # Load the XGBoost model from the downloaded file
        xgb_model = xgb.Booster()
        xgb_model.load_model(local_model_path)
        
        # Save the XGBoost model as a .pkl file using joblib
        joblib.dump(xgb_model, os.path.join(output_dir, "model.pkl"))
        print("Standalone XGBoost model saved as model.pkl.")

    # Check if the model is scikit-learn
    elif "python_function" in flavors and flavors["python_function"]["loader_module"] == "mlflow.sklearn":
        print("Detected scikit-learn model.")
        # Access the scikit-learn model
        sklearn_model = loaded_model._model_impl

        # Save the scikit-learn model as a .pkl file using joblib
        joblib.dump(sklearn_model, os.path.join(output_dir, "model.pkl"))
        print("Standalone scikit-learn model saved as model.pkl.")

def main(args):
    model_version = 1

    client = get_mlflow_client()
    model_name = "lead_model"

    def wait_for_deployment(model_name, model_version, stage='Staging'):
        status = False
        while not status:
            model_version_details = dict(
                client.get_model_version(name=model_name,version=model_version)
                )
            if model_version_details['current_stage'] == stage:
                status = True
                break
            else:
                time.sleep(2)
        return status

    model_version_details = dict(client.get_model_version(name=model_name,version=model_version))
    if model_version_details['current_stage'] != 'Staging':
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,stage="Staging", 
            archive_existing_versions=True
        )
        wait_for_deployment(model_name, model_version, 'Staging')
    else:
        print('Model already in staging')

    # Save model so it can go through inference testing
    model_uri = f"models:/{model_name}/Staging"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Specify output directory for saving the model
    output_dir = args.models_dir
    save_model(loaded_model, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--models_dir", type=str, required=True, help="Path to save models")

    args = parser.parse_args()
    main(args)
