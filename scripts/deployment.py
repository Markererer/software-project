import time
import argparse
from mlflow_client import get_mlflow_client
import mlflow
import joblib
import os
import atexit
import sys

class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

def save_model(model_uri, output_dir):
    try:
        print("Attempting to load as an XGBoost model...")
        xgb_model = mlflow.xgboost.load_model(model_uri=model_uri)
        joblib.dump(xgb_model, os.path.join(output_dir, "model.pkl"))
        print("Standalone XGBoost model saved as model.pkl.")
    except Exception as e:
        print(f"Not an XGBoost model: {e}")
        try:
            print("Attempting to load as a scikit-learn model...")
            lr_model = mlflow.sklearn.load_model(model_uri=model_uri)
            joblib.dump(lr_wrapper(lr_model), os.path.join(output_dir, "model.pkl"))
            print("Standalone Logistic Regression model saved as model.pkl.")
        except Exception as e:
            print(f"Could not load model: {e}")
            raise # Fail, print

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