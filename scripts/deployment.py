import time
import argparse
from mlflow_client import get_mlflow_client
import mlflow
import joblib
import os
from lr_wrapper import lr_wrapper

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
    joblib.dump(loaded_model, os.path.join(args.models_dir, "model.pkl"))
    print("Model has been saved as model.pkl.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--models_dir", type=str, required=True, help="Path to save models")

    args = parser.parse_args()
    main(args)
