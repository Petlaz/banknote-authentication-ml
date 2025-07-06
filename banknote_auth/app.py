#app.py
import gradio as gr
import joblib
import numpy as np

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict(Variance_Wavelet, Skewness_Wavelet, Curtosis_Wavelet, Image_Entropy):
    features = np.array([[Variance_Wavelet, Skewness_Wavelet, Curtosis_Wavelet, Image_Entropy]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    return "Authentic" if prediction[0] == 1 else "Forged"

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


# To run the Gradio app, execute: python hf_space/app.py
# Then open the provided link in your browser to interact with the model.
# Make sure you have the model and scaler files in the "models" directory.