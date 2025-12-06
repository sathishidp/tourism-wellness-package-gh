"""
register_model.py
- Uploads the trained pipeline artifact and metrics to the Hugging Face Model Hub.
- Writes a simple README (model card) including metrics and best params (if present).
- Non-interactive: uses HF_TOKEN env var.
"""
import os
import json
from pathlib import Path
from huggingface_hub import create_repo, upload_file, HfApi

HF_USERNAME = os.getenv("HF_USERNAME", "sathishaiuse")
MODEL_NAME = os.getenv("HF_MODEL_NAME", "wellness-tourism-model-best")
MODEL_REPO = f"{HF_USERNAME}/{MODEL_NAME}"

ARTIFACTS_DIR = Path("tourism_project/model_building/artifacts")
BEST_PIPE = ARTIFACTS_DIR / "best_pipeline.joblib"
METRICS_JSON = ARTIFACTS_DIR / "best_metrics.json"
STUDY_PKL = ARTIFACTS_DIR / "optuna_study.pkl"
README_PATH = ARTIFACTS_DIR / "README.md"

def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not found in environment.")

    if not BEST_PIPE.exists():
        raise FileNotFoundError(f"{BEST_PIPE} not found. Run train_and_tune.py first.")

    api = HfApi()
    try:
        create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
        print("Model repo ready:", MODEL_REPO)
    except Exception as e:
        print("Warning creating repo:", e)

    # Build model card content
    metrics_obj = None
    if METRICS_JSON.exists():
        try:
            metrics_obj = json.load(open(METRICS_JSON, "r"))
        except Exception:
            metrics_obj = None

    model_card = "# Wellness Tourism — Best Model\n\n"
    model_card += f"**Model repository:** {MODEL_REPO}\n\n"
    model_card += "**Trained with:** Optuna hyperparameter tuning and MLflow tracking.\n\n"
    if metrics_obj:
        model_card += "## Evaluation metrics (test set)\n\n"
        model_card += "```\n" + json.dumps(metrics_obj, indent=2) + "\n```\n\n"

    model_card += "## Usage\nDownload `best_pipeline.joblib` and load with `joblib.load()`.\n"

    # write README locally
    README_PATH.write_text(model_card)
    print("Wrote model card to:", README_PATH)

    # Upload files
    upload_targets = []
    upload_targets.append((str(BEST_PIPE), "best_pipeline.joblib"))
    if METRICS_JSON.exists():
        upload_targets.append((str(METRICS_JSON), "best_metrics.json"))
    upload_targets.append((str(README_PATH), "README.md"))

    for local_path, remote_name in upload_targets:
        try:
            print(f"Uploading {local_path} -> {MODEL_REPO}/{remote_name}")
            upload_file(path_or_fileobj=str(local_path), path_in_repo=remote_name, repo_id=MODEL_REPO, repo_type="model", token=token)
            print("Uploaded:", remote_name)
        except Exception as e:
            print("Upload failed for", remote_name, ":", type(e).__name__, e)

    print("Model registration complete. See:", f"https://huggingface.co/{MODEL_REPO}")

if __name__ == "__main__":
    main()
