import streamlit as tf
import pandas as pd
import pickle
from huggingface_hub import hf_hub_download

tf.set_page_config(page_title="Visit with Us - Package Prediction Engine", layout="centered")

tf.title(" Wellness Tourism Package Purchase Predictor")
tf.write("Input prospective customer interaction parameters beneath to calculate purchase probability scores *before* dispatching outreach calls.")

# 1. Coordinate model hub download references
USERNAME = "vivekvdeevi" 
MODEL_REPO = f"{USERNAME}/tourism-package-xgb-model"
MODEL_FILENAME = "tourism_xgb_pipeline.pkl"

@tf.cache_resource
def fetch_production_model_pipeline():
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as err:
        tf.error(f"Error fetching model registry from Hub: {err}")
        return None

pipeline = fetch_production_model_pipeline()

if pipeline:
    tf.success("Production Predictive Engine Loaded from Hugging Face Hub!")
    
    tf.subheader("Customer Profile Details")
    col1, col2 = tf.columns(2)
    
    with col1:
        age = tf.number_input("Customer Age", min_value=18, max_value=100, value=35)
        contact_type = tf.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
        city_tier = tf.selectbox("City Tier Category (1-3)", [1, 2, 3], index=0)
        gender = tf.selectbox("Gender", ["Male", "Female"])
        marital = tf.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        
    with col2:
        occupation = tf.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
        designation = tf.selectbox("Organization Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        income = tf.number_input("Gross Monthly Income", min_value=0, value=25000)
        trips = tf.number_input("Average Annual Trips", min_value=0, value=3)
        persons = tf.number_input("Total Persons Accompanying", min_value=1, value=2)
        children = tf.number_input("Number of Children Accompanying (< Age 5)", min_value=0, value=0)

    tf.subheader("Sales Pitch Interaction Details")
    col3, col4 = tf.columns(2)
    with col3:
        pitch_dur = tf.number_input("Pitch Duration (Minutes)", min_value=0, value=15)
        followups = tf.number_input("Scheduled Call Follow-ups", min_value=0, value=3)
        product_pitched = tf.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    with col4:
        prop_star = tf.selectbox("Preferred Hotel Property Star Rating", [3, 4, 5], index=0)
        passport = tf.radio("Holds Valid Passport?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        own_car = tf.radio("Owns Personal Vehicle?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        satisfie_score = tf.slider("Assumed Pitch Satisfaction Score", 1, 5, 3)

    # 2. Build structured DataFrame payload including ALL required structural columns
    input_payload = {
        "Unnamed: 0": 0,  # Mandatory tracking row token to align with model matrices
        "Age": age,
        "TypeofContact": contact_type,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "PreferredPropertyStar": prop_star,
        "MaritalStatus": marital,
        "NumberOfTrips": trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income,
        "PitchSatisfactionScore": satisfie_score,
        "ProductPitched": product_pitched,
        "NumberOfFollowups": followups,
        "DurationOfPitch": pitch_dur
    }
    
    if tf.button("Run Conversion Likelihood Score", type="primary"):
        features_df = pd.DataFrame([input_payload])
        
        # Enforce the exact sequence that matches your trained model schema
        expected_order = [
            "Unnamed: 0", "Age", "TypeofContact", "CityTier", "Occupation", "Gender",
            "NumberOfPersonVisiting", "PreferredPropertyStar", "MaritalStatus",
            "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
            "Designation", "MonthlyIncome", "PitchSatisfactionScore", 
            "ProductPitched", "NumberOfFollowups", "DurationOfPitch"
        ]
        features_df = features_df[expected_order]
        
        prediction = pipeline.predict(features_df)[0]
        probability = pipeline.predict_proba(features_df)[0][1]
        
        tf.markdown("---")
        tf.subheader("Predictive Analytics Output Assessment")
        
        score_percentage = probability * 100
        tf.metric(label="Calculated Purchase Probability", value=f"{score_percentage:.2f}%")
        
        if prediction == 1:
            tf.warning("Outcome: High Probability Purchaser! Prioritize customer profile for immediate marketing campaign engagement.")
        else:
            tf.info("Outcome: Low Conversion Probability. Route customer through standard passive marketing pipelines.")
else:
    tf.error("Pipeline compilation error.")
