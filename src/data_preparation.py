import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

def run_preparation():
    print("--- [STEP 2] Running Clean Data Preparation ---")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    token = os.getenv("HF_TOKEN")
    repo_id = f"{config['huggingface']['username']}/{config['huggingface']['dataset_repo']}"
    
    # FIXED WEB ROUTING PATH WITH FORWARD SLASHES INTEGRATED PERFECTLY
    hf_url = f"https://huggingface.co{repo_id}/resolve/main/tourism.csv"
    df = pd.read_csv(hf_url)
    
    df = df.drop(columns=[col for col in config['features']['drop_columns'] if col in df.columns], errors='ignore')
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
        
    for col in config['features']['numerical']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    for col in config['features']['categorical']:
        if col in df.columns:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
                
    train_df, test_df = train_test_split(
        df, test_size=config['data']['test_size'], random_state=config['data']['random_state'],
        stratify=df[config['data']['target_column']] if config['data']['target_column'] in df.columns else None
    )
    
    os.makedirs("data", exist_ok=True)
    train_df.to_csv(config['data']['train_local_path'], index=False)
    test_df.to_csv(config['data']['test_local_path'], index=False)
    
    api = HfApi()
    api.upload_file(path_or_fileobj=config['data']['train_local_path'], path_in_repo="train.csv", repo_id=repo_id, repo_type="dataset", token=token)
    api.upload_file(path_or_fileobj=config['data']['test_local_path'], path_in_repo="test.csv", repo_id=repo_id, repo_type="dataset", token=token)
    print("Data Preparation Complete Script Sync.")

if __name__ == "__main__":
    run_preparation()
