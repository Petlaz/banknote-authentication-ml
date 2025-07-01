# banknote_auth/features.py

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
from loguru import logger

from banknote_auth.config import PROCESSED_DATA_DIR, MODELS_DIR

def load_clean_data(filename="data_cleaned.csv") -> pd.DataFrame:
    data_path = PROCESSED_DATA_DIR / filename
    logger.info(f"Loading cleaned data from: {data_path}")
    return pd.read_csv(data_path)

def split_features_targets(df: pd.DataFrame, target_column: str = "Class"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    logger.info(f"Split data into X (shape: {X.shape}) and y (shape: {y.shape})")
    return X, y

def scale_features(X_train, X_test=None, scaler_path=MODELS_DIR / "scaler.pkl", save_scaler=True):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    logger.info("Features scaled with StandardScaler.")

    if save_scaler:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to: {scaler_path}")

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    return X_train_scaled
