import time
import argparse
from mlflow_client import get_mlflow_client
import mlflow
import joblib
import os
from lr_wrapper import lr_wrapper
import xgboost as xgb
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def save_model_as_pkl(loaded_model, output_dir):
    try:
        flavors = loaded_model.metadata.flavors

        # Check if the model is XGBoost
        if "xgboost" in flavors:
            logging.debug("Detected XGBoost model.")
            artifact_path = flavors["xgboost"]["data"]

            # Download the artifact to a local directory
            logging.debug("Downloading artifact...")
            local_model_path = mlflow.artifacts.download_artifacts(artifact_path)
            logging.debug(f"Downloaded artifact to: {local_model_path}")

            # Load the XGBoost model
            xgb_model = xgb.Booster()
            xgb_model.load_model(local_model_path)
            logging.debug("XGBoost model loaded successfully.")

            # Save the XGBoost model as .pkl
            joblib.dump(xgb_model, os.path.join(output_dir, "model.pkl"))
            logging.debug("XGBoost model saved as model.pkl.")

        # Check if the model is scikit-learn
        elif "python_function" in flavors and flavors["python_function"]["loader_module"] == "mlflow.sklearn":
            logging.debug("Detected scikit-learn model.")
            sklearn_model = loaded_model._model_impl

            # Save the scikit-learn model as .pkl
            joblib.dump(sklearn_model, os.path.join(output_dir, "model.pkl"))
            logging.debug("Scikit-learn model saved as model.pkl.")

        else:
            raise ValueError("Unsupported model type. Only XGBoost and scikit-learn models are supported.")

    except Exception as e:
        logging.error(f"Error in save_model_as_pkl: {e}", exc_info=True)
        raise

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

    try:
        logging.debug("Starting the script...")
        model_uri = f"models:/{model_name}/Staging"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        logging.debug(f"Loaded model from URI: {model_uri}")

        # Specify output directory
        output_dir = args.models_dir
        if not os.path.exists(output_dir):
            logging.debug(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        save_model(loaded_model, output_dir)
        logging.debug("Model export completed.")

    except Exception as e:
        logging.error(f"Script failed with error: {e}", exc_info=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--models_dir", type=str, required=True, help="Path to save models")

    args = parser.parse_args()
    main(args)
