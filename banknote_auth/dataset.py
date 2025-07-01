# banknote_auth/dataset.py
# -*- coding: utf-8 -*-
import os
import pandas as pd
import urllib.request

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
RAW_DATA_PATH = "data/raw/data_banknote_authentication.txt"


def prepare_data_folders(base_path="data"):
    folders = ["raw", "interim", "processed", "external"]
    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
    print("Data folders created or already exist.")


def download_dataset():
    if not os.path.exists(RAW_DATA_PATH):
        print("Downloading dataset from UCI...")
        urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
        print(f"Dataset saved to {RAW_DATA_PATH}")
    else:
        print("Dataset already downloaded.")


def load_dataset():
    column_names = [
        "Variance_Wavelet",
        "Skewness_Wavelet",
        "Curtosis_Wavelet",
        "Image_Entropy",
        "Class"
    ]
    df = pd.read_csv(RAW_DATA_PATH, header=None, names=column_names)
    return df


if __name__ == "__main__":
    prepare_data_folders()
    download_dataset()
    df = load_dataset()
    print("Dataset preview:")
    print(df.head())
