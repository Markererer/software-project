import subprocess

def main():
    subprocess.run(["dvc", "pull"], check=True)

    print(f"Success!")

if __name__ == "__main__":
    main()
