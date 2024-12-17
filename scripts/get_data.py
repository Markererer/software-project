import os
import subprocess
import shutil
import argparse

def main(args):
    # Repository URL and file paths
    repo_url = "https://github.com/Jeppe-T-K/itu-sdse-project-data"
    file_path = "raw_data.csv"  # The file we want from the repo
    local_repo_dir = "itu-sdse-project-data"
    destination_path = os.path.join(args.raw_data_dir, "raw_data.csv")

    # Step 1: Clone the repository (if not already cloned)
    if not os.path.exists(local_repo_dir):
        # Make sure you have a recent git or remove unsupported filters
        subprocess.run(["git", "clone", "--depth", "1", repo_url], check=True)

    # Step 2: Pull the DVC-tracked file(s)
    # Run from within the cloned repo directory
    subprocess.run(["dvc", "pull"], cwd=local_repo_dir, check=True)

    # Step 3: Move the file to the desired local path
    destination_dir = os.path.dirname(destination_path)
    os.makedirs(destination_dir, exist_ok=True)

    # Move the file (it should now exist after a successful dvc pull)
    shutil.move(os.path.join(local_repo_dir, file_path), destination_path)

    print(f"{file_path} has been pulled and saved to {destination_path} successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Download Script")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to the raw data directory")
    args = parser.parse_args()
    main(args)
