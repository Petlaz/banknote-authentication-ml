# deployment/app_gradio.py
import gradio as gr
import joblib
import numpy as np
from pathlib import Path

#Smart path handling that works from both locations
current_dir = Path(__file__).parent
model_path = current_dir.parent/"models"/"best_model.pkl"  # Goes up one level
scaler_path = current_dir.parent/"models"/"scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
def predict(Variance_Wavelet, Skewness_Wavelet, Curtosis_Wavelet, Image_Entropy):
    features = np.array([[Variance_Wavelet, Skewness_Wavelet, Curtosis_Wavelet, Image_Entropy]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    return "Authentic" if prediction[0] == 1 else "Forged"

# Define Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Variance of Wavelet"),
        gr.Number(label="Skewness of Wavelet"),
        gr.Number(label="Curtosis of Wavelet"),
        gr.Number(label="Entropy of Image")
    ],
    outputs="text",
    title="Banknote Authentication",
    description="Enter the banknote features to check if it's authentic or forged."
)

if __name__ == "__main__":
    interface.launch(share=True)

# To run the Gradio app, execute: python deployment/app_gradio.py  
# Then open the provided link in your browser to interact with the model.
# Make sure you have the model and scaler files in the "models" directory.
