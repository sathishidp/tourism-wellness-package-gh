"""
push_to_space.py
- Pushes all deployment files from tourism_project/deployment/ to the Hugging Face Space.
- Fully non-interactive: reads HF_TOKEN from environment.
- Designed for GitHub Actions or local scripted use.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
HF_USERNAME = os.getenv("HF_USERNAME", "sathishaiuse")
SPACE_NAME = os.getenv("HF_SPACE_NAME", "wellness-tourism-deployment")
SPACE_REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"

HF_TOKEN = os.getenv("HF_TOKEN")  # MUST be provided in GH Actions
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not found. Set HF_TOKEN environment variable.")

DEPLOY_DIR = Path("tourism_project/deployment")

def main():
    print(f"Using HF token: {HF_TOKEN[:4]}... (hidden)")  # Mask token for logs

    api = HfApi()

    # ---------------------------------------------------
    # Create or reuse Space repo
    # ---------------------------------------------------
    print(f"Creating or reusing Hugging Face Space: {SPACE_REPO_ID}")
    try:
        create_repo(
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            exist_ok=True,
            token=HF_TOKEN
        )
        print(f"Space is ready: https://huggingface.co/spaces/{SPACE_REPO_ID}")
    except Exception as e:
        print("Warning: Space repo creation issue:", e)

    # ---------------------------------------------------
    # Upload files from deployment folder
    # ---------------------------------------------------
    print("\nUploading deployment files...\n")

    if not DEPLOY_DIR.exists():
        raise FileNotFoundError(f"Deployment folder not found: {DEPLOY_DIR}")

    for file_path in DEPLOY_DIR.iterdir():
        if file_path.is_file():
            try:
                print(f"Uploading: {file_path.name}")
                upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=SPACE_REPO_ID,
                    repo_type="space",
                    token=HF_TOKEN
                )
                print(f"Uploaded: {file_path.name}\n")
            except Exception as e:
                print(f"Failed to upload {file_path.name}: {e}\n")

    print("Deployment complete!")
    print(f"Visit your Space at: https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    main()
