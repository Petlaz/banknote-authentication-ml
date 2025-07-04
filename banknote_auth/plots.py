# banknote_auth/plots.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from pathlib import Path

from banknote_auth.config import PROJ_ROOT

# Define a consistent style
sns.set(style="whitegrid")

def save_confusion_matrix(cm, labels, title="Confusion Matrix", filename="confusion_matrix.png"):
    """Save a confusion matrix heatmap to the reports/figures directory."""
    fig_dir = PROJ_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)

    file_path = fig_dir / filename
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {file_path}")


def show_confusion_matrix(cm, labels, title="Confusion Matrix"):
    """Show a confusion matrix inline (e.g., in Jupyter)."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()


def plot_roc_curve(model, X_test, y_test, title="ROC Curve"):
    """Display ROC curve (inline)"""
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(title)
    plt.show()
