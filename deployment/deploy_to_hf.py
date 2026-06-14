import os
import yaml
from huggingface_hub import HfApi

def push_to_space():
    print("--- [AUTOMATED STEP 4] Pushing Application Files to Hugging Face Spaces ---")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Dynamically pull the secure secret key token from the runner environment settings
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("CRITICAL ERROR: HF_TOKEN variable is missing inside the environment runner workspace!")
        
    space_id = f"{config['huggingface']['username']}/{config['huggingface']['space_repo']}"
    
    api = HfApi()
    print(f"Initializing HF space destination workspace: {space_id}")
    api.create_repo(repo_id=space_id, token=token, repo_type="space", space_sdk="docker", exist_ok=True)
    
    deployment_files = {
        "deployment/app.py": "app.py",
        "deployment/Dockerfile": "Dockerfile",
        "deployment/requirements.txt": "requirements.txt"
    }
    
    for local_path, repo_path in deployment_files.items():
        print(f"Uploading deployment asset element {local_path} --> {repo_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=space_id,
            repo_type="space",
            token=token
        )
    print(" Production application files successfully synced up to Hugging Face Spaces.\n")

if __name__ == "__main__":
    push_to_space()
