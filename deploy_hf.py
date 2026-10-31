"""
One-click deployment script to Hugging Face Spaces (Free Cloud Hosting with Docker).
"""

import os
import sys
from huggingface_hub import HfApi

SPACE_README = """---
title: CreditIQ Bank Credit Scoring
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# CreditIQ: Bank Credit Risk & Decision Engine
Interactive Credit Default Predictor & Automated Underwriting System.
"""


def deploy(repo_id: str, token: str):
    """
    Creates a new Docker-based Space on Hugging Face and uploads all project files.
    """
    api = HfApi(token=token)
    project_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Creating Hugging Face Space: {repo_id}...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True
        )
        print(f"Space '{repo_id}' created or verified on Hugging Face.")
    except Exception as e:
        print(f"Space creation error: {e}")
        raise e

    # Write README.md with space metadata
    hf_readme_path = os.path.join(project_dir, "README_SPACE.md")
    with open(hf_readme_path, "w") as f:
        f.write(SPACE_README)

    print("Uploading README.md configuration...")
    api.upload_file(
        path_or_fileobj=hf_readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space"
    )

    print("Uploading project files (models, code, requirements, dataset)...")
    api.upload_folder(
        folder_path=project_dir,
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[
            ".venv/*",
            ".git/*",
            "__pycache__/*",
            "*.pyc",
            "README_SPACE.md",
            "deploy_hf.py"
        ]
    )

    if os.path.exists(hf_readme_path):
        os.remove(hf_readme_path)

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print("=" * 60)
    print(f"🎉 SUCCESS! Your application is deploying live at:")
    print(f"👉 {space_url}")
    print("=" * 60)
    return space_url


if __name__ == "__main__":
    if len(sys.argv) < 3:
        token = os.environ.get("HF_TOKEN")
        repo_id = os.environ.get("HF_REPO_ID")
        if not token or not repo_id:
            print("Usage: uv run python deploy_hf.py <username/space-name> <hf_write_token>")
            sys.exit(1)
    else:
        repo_id = sys.argv[1]
        token = sys.argv[2]

    deploy(repo_id, token)
