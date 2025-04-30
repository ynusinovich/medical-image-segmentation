# scripts/download_data.py

import os
import tarfile

import gdown


def download_and_extract_liver_data(destination_folder="data/raw"):
    # Google Drive file ID for Task03_Liver.tar
    file_id = "1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu"
    url = f"https://drive.google.com/uc?id={file_id}"

    # Ensure destination directory exists
    os.makedirs(destination_folder, exist_ok=True)

    tar_path = os.path.join(destination_folder, "Task03_Liver.tar")

    # Download if not already downloaded
    if not os.path.exists(tar_path):
        print(f"Downloading Task03_Liver.tar to {tar_path}...")
        gdown.download(url, tar_path, quiet=False)
        print("Download complete.")
    else:
        print(f"File already exists at {tar_path}. Skipping download.")

    # Extract
    extract_folder = os.path.join(destination_folder, "Task03_Liver")
    if not os.path.exists(extract_folder):
        print(f"Extracting {tar_path} to {extract_folder}...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=destination_folder)
        print("Extraction complete.")
    else:
        print(
            f"Extraction folder {extract_folder} already exists. Skipping extraction."
        )


if __name__ == "__main__":
    download_and_extract_liver_data()
