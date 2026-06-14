import os
import pickle
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from huggingface_hub import HfApi

def run_modeling_pipeline():
    print("--- [STEP 3] Running Automated Model Building & Registry Tracking ---")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    token = os.getenv("HF_TOKEN")
    dataset_repo = f"{config['huggingface']['username']}/{config['huggingface']['dataset_repo']}"
    model_repo = f"{config['huggingface']['username']}/{config['huggingface']['model_repo']}"
    
    # FIXED WEB ROUTING PATH FOR CLOUD ACTIONS
    train_url = f"https://huggingface.co{dataset_repo}/resolve/main/train.csv"
    test_url = f"https://huggingface.co{dataset_repo}/resolve/main/test.csv"
    
    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)
    
    X_train = train_df.drop(columns=[config['data']['target_column']])
    y_train = train_df[config['data']['target_column']]
    X_test = test_df.drop(columns=[config['data']['target_column']])
    y_test = test_df[config['data']['target_column']]
    
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)], 
        remainder='passthrough'
    )
    
    negative_class_count = len(y_train) - sum(y_train)
    imbalance_scale_ratio = negative_class_count / sum(y_train)
    
    tuned_params = {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.08,
        "scale_pos_weight": float(imbalance_scale_ratio),
        "random_state": int(config['data']['random_state']),
        "eval_metric": "logloss"
    }
    
    clf = xgb.XGBClassifier(**tuned_params)
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    model_pipeline.fit(X_train, y_train)
    
    model_local_file = "tourism_xgb_pipeline.pkl"
    with open(model_local_file, "wb") as m_file:
        pickle.dump(model_pipeline, m_file)
        
    api = HfApi()
    api.create_repo(repo_id=model_repo, token=token, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=model_local_file, path_in_repo="tourism_xgb_pipeline.pkl", repo_id=model_repo, repo_type="model", token=token)
    print(" Automated Model Pipeline Registry Sync Complete.")

if __name__ == "__main__":
    run_modeling_pipeline()
