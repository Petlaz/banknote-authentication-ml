# deployment/app_gradio.py
import gradio as gr
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

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
    interface.launch()
