"""Put the project online for free on Hugging Face.

Before running:  create an account at huggingface.co, make a token with WRITE access
(Settings -> Access Tokens), then run:

    python deploy.py

It uploads your trained model, creates a public web app (a "Space"), and prints the link.
"""

import os
import sys

from huggingface_hub import HfApi, get_token, login

from src import config

MODEL_REPO_NAME = "code-comment-generator"
SPACE_NAME = "code-comment-generator-demo"


def main():
    if not os.path.isfile(os.path.join(config.MODEL_DIR, "config.json")):
        sys.exit(f"No trained model in '{config.MODEL_DIR}'. Train first: python -m src.training")

    if not get_token():
        login()                                  # asks you to paste the token once
    api = HfApi()
    user = api.whoami()["name"]
    model_repo = f"{user}/{MODEL_REPO_NAME}"
    space_repo = f"{user}/{SPACE_NAME}"

    print(f"Uploading model to {model_repo} ...")
    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=config.MODEL_DIR, repo_id=model_repo, repo_type="model")

    print(f"Creating web app {space_repo} ...")
    api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_variable(space_repo, "MODEL_ID", model_repo)   # tells app.py which model to load
    api.upload_file(path_or_fileobj="app.py", path_in_repo="app.py", repo_id=space_repo, repo_type="space")
    api.upload_file(path_or_fileobj="space_requirements.txt", path_in_repo="requirements.txt",
                    repo_id=space_repo, repo_type="space")
    api.upload_folder(folder_path="src", path_in_repo="src", repo_id=space_repo, repo_type="space",
                      allow_patterns=["*.py"])

    print("\nDone.")
    print(f"Model   : https://huggingface.co/{model_repo}")
    print(f"Live app: https://huggingface.co/spaces/{space_repo}")
    print("The app needs a few minutes to build the first time. Watch progress on that page.")


if __name__ == "__main__":
    main()
