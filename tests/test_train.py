import pytest
import joblib
import numpy as np
from banknote_auth.features import load_clean_data, split_features_targets, scale_features
from sklearn.model_selection import train_test_split
from pathlib import Path

MODELS_DIR = Path("models")

@pytest.fixture(scope="module")
def trained_model():
    # You should have a function that trains and saves the model, e.g. train_model()
    # from banknote_auth.modeling.train import train_model
    # train_model()
    model_path = MODELS_DIR / "best_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    return joblib.load(model_path), joblib.load(scaler_path)

def test_model_and_scaler_exist():
    assert (MODELS_DIR / "best_model.pkl").exists(), "Model file missing"
    assert (MODELS_DIR / "scaler.pkl").exists(), "Scaler file missing"

def test_model_prediction_binary(trained_model):
    model, scaler = trained_model
    df = load_clean_data()
    X, y = split_features_targets(df)
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    assert set(preds).issubset({0, 1}), "Predictions not binary"

def test_model_has_predict_method(trained_model):
    model, _ = trained_model
    assert hasattr(model, "predict"), "Model missing predict() method"

# To verify training and prediction pipeline:  pytest tests/test_train.py