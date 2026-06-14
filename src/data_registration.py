import os
import yaml
from huggingface_hub import HfApi

def run_registration():
    print("--- [STEP 1] Starting Data Registration ---")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("CRITICAL ERROR: HF_TOKEN environment variable is missing!")

    api = HfApi()
    repo_id = f"{config['huggingface']['username']}/{config['huggingface']['dataset_repo']}"
    
    print(f"Checking or creating dataset repository on Hugging Face: {repo_id}")
    api.create_repo(repo_id=repo_id, token=token, repo_type="dataset", exist_ok=True)
    
    print(f"Uploading file {config['data']['raw_local_path']} as tourism.csv...")
    api.upload_file(
        path_or_fileobj=config['data']['raw_local_path'],
        path_in_repo="tourism.csv",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    print("Data Registration completed successfully.\n")

if __name__ == "__main__":
    run_registration()
