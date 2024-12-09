import os
import subprocess
import shutil

# Repository URL and file path
repo_url = "https://github.com/Jeppe-T-K/itu-sdse-project-data"
file_path = "itu-sdse-project-data/itu-sdse-project-main/data/raw/raw_data.csv"  # Path of the specific file in the repo
destination_path = "itu-sdse-project-main/data/raw/raw_data.csv"  # Desired local destination

# Step 1: Clone the repository (if not already cloned)
if not os.path.exists("itu-sdse-project-data"):
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", repo_url], shell=True)

# Step 2: Change directory to the repository
os.chdir("itu-sdse-project-data")

# Step 3: Pull the specific file using DVC
subprocess.run(["dvc", "pull", file_path], shell=True)

# Step 4: Move the file to the desired local path
# Ensure the destination directory exists
destination_dir = os.path.dirname(destination_path)
os.makedirs(destination_dir, exist_ok=True)

# Move the file
shutil.move(file_path, destination_path)

# Step 5: Optional - Provide confirmation
print(f"{file_path} has been pulled and saved to {destination_path} successfully!")

def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.
    
    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    import stat
    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

shutil.rmtree("itu-sdse-project-data", onerror=onerror)
