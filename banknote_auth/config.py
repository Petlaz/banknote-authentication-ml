# banknote_auth/config.py

from pathlib import Path
from dotenv import load_dotenv
import os

# Set project root
PROJ_ROOT = Path(__file__).resolve().parents[1]
print(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Load .env variables
load_dotenv(PROJ_ROOT / ".env")

# Define data directories
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# ✅ Add this to fix your error
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
