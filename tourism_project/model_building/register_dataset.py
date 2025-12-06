
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file

HF_USERNAME = os.getenv("HF_USERNAME", "sathishaiuse")
DATASET_NAME = os.getenv("HF_DATASET", "wellness-tourism-dataset-final")
REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"

LOCAL_PATH = Path("tourism_project/data/tourism.csv")

def main():
    if not LOCAL_PATH.exists():
        raise FileNotFoundError(f"Expected dataset at {LOCAL_PATH} (upload it first).")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not found in environment. Set it as a secret.")

    api = HfApi()
    try:
        create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
        print(f"Dataset repo ready: {REPO_ID}")
    except Exception as e:
        print("Warning creating dataset repo:", e)

    try:
        print(f"Uploading {LOCAL_PATH} to dataset repo {REPO_ID} ...")
        upload_file(
            path_or_fileobj=str(LOCAL_PATH),
            path_in_repo="tourism.csv",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token
        )
        print("Upload successful.")
    except Exception as e:
        print("Upload failed:", type(e).__name__, e)
        raise

if __name__ == "__main__":
    main()
