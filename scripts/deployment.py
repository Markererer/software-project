import time
import argparse
from mlflow_client import get_mlflow_client

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
    model_status = True #Unused?
    if model_version_details['current_stage'] != 'Staging':
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,stage="Staging", 
            archive_existing_versions=True
        )
        model_status = wait_for_deployment(model_name, model_version, 'Staging') #Unused?
    else:
        print('Model already in staging')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--artifacts_dir", type=str, required=True, help="Path to save artifacts")

    args = parser.parse_args()
    main(args)
