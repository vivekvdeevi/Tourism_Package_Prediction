import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi

def run_preparation():
    print("--- [STEP 2] Running Clean Data Preparation ---")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    token = os.getenv("HF_TOKEN")
    repo_id = f"{config['huggingface']['username']}/{config['huggingface']['dataset_repo']}"
    
    print(f"Downloading raw dataset securely from Hugging Face repository: {repo_id}")
    local_download_path = hf_hub_download(
        repo_id=repo_id,
        filename="tourism.csv",
        repo_type="dataset",
        token=token
    )
    
    df = pd.read_csv(local_download_path)
    
    # Explicit list definitions to entirely bypass any YAML dictionary parsing errors
    drop_columns = ["CustomerID"]
    numerical_features = [
        "Unnamed: 0", "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", 
        "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport", 
        "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"
    ]
    categorical_features = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]
    target_column = "ProdTaken"
    
    # 1. Clean unique identifiers
    df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')
    
    # 2. Standardise categorical text variances
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
        
    # 3. Safe Statistical Imputation Loop
    print("Imputing missing record fields...")
    for col in numerical_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    for col in categorical_features:
        if col in df.columns:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
                
    # 4. Stratified Split Partitioning
    print("Partitioning rows into stratified train/test arrays...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df[target_column] if target_column in df.columns else None
    )
    
    # Save files locally inside the required subfolders
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    
    # 5. Push Clean Data Splits back to Hugging Face
    print("Uploading clean matrices back to Hugging Face...")
    api = HfApi()
    api.upload_file(path_or_fileobj="data/train.csv", path_in_repo="train.csv", repo_id=repo_id, repo_type="dataset", token=token)
    api.upload_file(path_or_fileobj="data/test.csv", path_in_repo="test.csv", repo_id=repo_id, repo_type="dataset", token=token)
    print("🎉 Data Preparation Complete Script Sync.")

if __name__ == "__main__":
    run_preparation()
