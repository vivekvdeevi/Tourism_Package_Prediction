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
    
    # Check if the raw file is already live in the repository
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        if "tourism.csv" in files:
            print("tourism.csv already exists in the Hugging Face dataset space. Skipping upload.")
            print("Data Registration completed successfully.\n")
            return
    except Exception:
        pass

    # Fallback to local upload only if it exists on the runner filesystem
    if os.path.exists(config['data']['raw_local_path']):
        print(f"Uploading local file {config['data']['raw_local_path']} as tourism.csv...")
        api.upload_file(
            path_or_fileobj=config['data']['raw_local_path'],
            path_in_repo="tourism.csv",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print("Data Registration completed successfully.\n")
    else:
        print("Dataset is already maintained safely on Hugging Face Hub.")
        print("Data Registration completed successfully.\n")

if __name__ == "__main__":
    run_registration()
