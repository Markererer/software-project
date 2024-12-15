import time
import argparse
from mlflow_client import get_mlflow_client
import mlflow
import joblib
import os
from lr_wrapper import lr_wrapper
import xgboost as xgb
import atexit
import sys

def save_model(model_uri, output_dir):
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    # Get the flavors metadata
    flavors = loaded_model.metadata.flavors

    # Check if the model is XGBoost
    if "xgboost" in flavors:
        print("Detected XGBoost model.")
        xgb_model = mlflow.xgboost.load_model(model_uri=model_uri)

        joblib.dump(xgb_model, os.path.join(output_dir, "model.pkl"))
        print("Standalone XGBoost model saved as model.pkl.")

    # Check if the model is scikit-learn
    elif "python_function" in flavors and flavors["python_function"]["loader_module"] == "mlflow.sklearn":
        print("Detected scikit-learn model.")
        # Access the scikit-learn model
        sklearn_model = mlflow.sklearn.load_model(model_uri=model_uri)

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

    # Specify output directory for saving the model
    output_dir = args.models_dir
    save_model(model_uri, output_dir)

def exit_handler():
    print("Script exited unexpectedly!")
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--models_dir", type=str, required=True, help="Path to save models")

    args = parser.parse_args()
    atexit.register(exit_handler)
    print("Script is running with exit handler.")
    sys.stdout.flush()
    main(args)