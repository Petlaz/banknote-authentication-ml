# deployment/app_streamlit.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime

# Optional: Load logo
try:
    from PIL import Image
    logo = Image.open("assets/logo.png")
    st.image(logo, width=120)
except:
    st.markdown("### Banknote Authentication App")

# Load the trained model and scaler
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Streamlit inputs
st.title("🔍 Banknote Authentication")

st.markdown("Enter the features of the banknote:")

variance = st.number_input("Variance of Wavelet Transformed Image", value=0.0)
skewness = st.number_input("Skewness of Wavelet Transformed Image", value=0.0)
curtosis = st.number_input("Curtosis of Wavelet Transformed Image", value=0.0)
entropy = st.number_input("Entropy of Image", value=0.0)

def log_prediction(features, prediction):
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/prediction_log.csv"
    log_entry = {
        "Timestamp": datetime.now().isoformat(),
        "Variance": features[0][0],
        "Skewness": features[0][1],
        "Curtosis": features[0][2],
        "Entropy": features[0][3],
        "Prediction": prediction
    }
    df = pd.DataFrame([log_entry])
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, mode='w', header=True, index=False)

if st.button("Check Authenticity"):
    features = np.array([[variance, skewness, curtosis, entropy]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    result = "✅ Authentic" if prediction[0] == 1 else "❌ Forged"
    st.success(f"Prediction: {result}")

    log_prediction(features, result)
