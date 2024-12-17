import subprocess

def main():
    subprocess.run(["dvc", "pull"], check=True)

    print(f"DVC Pull success!")

if __name__ == "__main__":
    main()
