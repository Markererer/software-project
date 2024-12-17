import subprocess

def main():
    subprocess.run(["dvc", "update", "./data/raw/raw_data.csv.dvc"], check=True)
    subprocess.run(["dvc", "pull"], check=True)

    print(f"DVC Pull success!")

if __name__ == "__main__":
    main()
