from sklearn.model_selection import train_test_split
from banknote_auth.features import load_clean_data, split_features_targets, scale_features
from banknote_auth.modeling.models import build_voting_classifier
from banknote_auth.config import MODELS_DIR
import joblib

if __name__ == "__main__":
    # Load and split
    df = load_clean_data()
    X, y = split_features_targets(df)
    
    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Scale
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Train model
    model = build_voting_classifier()
    model.fit(X_train_scaled, y_train)

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the trained model
    joblib.dump(model, MODELS_DIR / "voting_classifier.pkl")
    print(f"✅ Model saved to {MODELS_DIR / 'voting_classifier.pkl'}")