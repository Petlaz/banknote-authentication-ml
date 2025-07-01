# banknote_auth/modeling/predict.py
import joblib
import numpy as np
import pandas as pd  # <-- Add this line

# Load scaler and model
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/best_model.pkl")

def predict_note_authentication(features):
    columns = [
        "Variance_Wavelet",
        "Skewness_Wavelet",
        "Curtosis_Wavelet",
        "Image_Entropy"
    ]
    features_df = pd.DataFrame([features], columns=columns)
    scaled_features = scaler.transform(features_df)
    prediction = model.predict(scaled_features)
    return "Authentic" if prediction[0] == 1 else "Forged"

# Example usage
if __name__ == "__main__":
    test_input = [3.6216, 8.6661, -2.8073, -0.44699]
    result = predict_note_authentication(test_input)
    print(f"Prediction: {result}")
    
## You can run it to test the prediction function using this code block.
# python -m banknote_auth.modeling.predict