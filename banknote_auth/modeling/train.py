import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from loguru import logger
import os
from banknote_auth.config import PROCESSED_DATA_DIR

csv_path = PROCESSED_DATA_DIR / "data_cleaned.csv"

from banknote_auth.config import (
    CSV_PATH,
    MODELS_DIR,
    BEST_MODEL_PATH,
    SCALER_PATH,
)

logger.info(f"Using cleaned dataset from: {CSV_PATH}")

# Verify the file exists before attempting to load
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Cleaned CSV not found at: {CSV_PATH}")

# Load dataset
data_cleaned = pd.read_csv(csv_path)
logger.info(f"Dataset loaded with shape: {data_cleaned.shape}")



# Feature and label separation
X = data_cleaned.iloc[:, :-1]
y = data_cleaned.iloc[:, -1]


logger.info(f"Target distribution:\n{y.value_counts()}")
logger.info(f"X shape: {X.shape}, y shape: {y.shape}")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, SCALER_PATH)
logger.info(f"Scaler saved to: {SCALER_PATH}")

# Define base models
svm = SVC(probability=True, random_state=42)
#knn = KNeighborsClassifier()
knn = KNeighborsClassifier(n_neighbors=1)
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Ensemble voting classifier
voting_model = VotingClassifier(
    estimators=[("svm", svm), ("knn", knn), ("rf", rf), ("xgb", xgb)],
    voting="soft"
)

# Train ensemble model
logger.info("Training ensemble Voting Classifier...")
voting_model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(voting_model, BEST_MODEL_PATH)
logger.success(f"Model training complete. Saved to {BEST_MODEL_PATH}")

# Evaluate
accuracy = voting_model.score(X_test_scaled, y_test)
logger.info(f"Model accuracy on test set: {accuracy:.4f}")
