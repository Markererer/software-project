import subprocess
import argparse
import os

def main(args):
    subprocess.run(["dvc", "update", os.path.join(args.raw_data_dir, "raw_data.csv.dvc")], check=True)
    subprocess.run(["dvc", "pull"], check=True)

    print(f"DVC Pull success!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Downloading Script")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to the raw data directory")

    args = parser.parse_args()
    main(args)
