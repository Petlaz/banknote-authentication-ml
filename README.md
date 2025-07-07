---
title: banknote_authentication

app_file: deployment/app_gradio.py

sdk: gradio

sdk_version: 5.35.0

license: mit

emoji: 👁

colorFrom: yellow

colorTo: red

pinned: false

short_description: A machine learning project to classify banknotes as genuine or forged.
---

# Banknote Authentication with Machine Learning

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A machine learning project to classify banknotes as genuine or forged using statistical features extracted from images.


## Data Source

- [UCI ML Repository: Banknote Authentication Data Set](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)


## Project Structure

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         banknote_auth and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── banknote_auth   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes banknote_auth a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Quickstarts

1. **Clone the repository**

2. **Install dependencies**

pip install -r requirements.txt

3. **Train the model**

python -m banknote_auth.modeling.train

4. **Generate report**

python -m banknote_auth.reporting.generate_report

5. **Run prediction**

python -m banknote_auth.modeling.predict

6. **Run Test**

pytest tests/

7. **Run Gradio App**

python deployment/app_gradio.py


## Model Details

* **Algorithm: VotingClassifier (ensemble of Random Forest, XGBoost, SVM, KNN)**

* **Features Used: Variance, Skewness, Curtosis, Entropy**

* **Scaling: StandardScaler**

* **Test Size: 20%**


## Visualizations

Here’s the confusion matrix from the final model:

![Confusion Matrix](reports/figures/confusion_matrix.png)

[![ROC Curve](reports/figures/roc_curve.png)](reports/figures/roc_curve.png)

Accuracy and classification report are saved in:
- [`reports/metrics/classification_report.txt`](reports/metrics/classification_report.txt)

- [`reports/metrics/classification_report.json`](reports/metrics/classification_report.json)


## Model Metrics

**Test Accuracy:** `1.00`

**Classification Report:**

**Precision, Recall, F1-score:**  
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       135
           1       1.00      1.00      1.00        91

    accuracy                           1.00       226
   macro avg       1.00      1.00      1.00       226
weighted avg       1.00      1.00      1.00       226
```

## Deployment

🟢 Local Deployment

1. **Gradio App (Local)**

python deployment/app_gradio.py

Launches the Gradio web interface in your browser.

☁️ Cloud Deployment

1. **Hugging Face Spaces**

📍 Live Demo: https://huggingface.co/spaces/petlaz/banknote-authentication


## Sample Prediction

import joblib

import numpy as np

model = joblib.load("models/best_model.pkl")

scaler = joblib.load("models/scaler.pkl")

sample = np.array([[2.3, 6.7, -1.2, 0.5]])

scaled = scaler.transform(sample)

result = model.predict(scaled)

print("Prediction:", "Authentic" if result[0] == 0 else "Forged")


## References

1. **UCI Banknote Dataset**

2. **Géron, A. Hands-On Machine Learning with Scikit-Learn and TensorFlow**

3. **Ng, A. Machine Learning Specialization, Coursera**

4. **Gradio Documentation**


## Requirements
- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- (see `requirements.txt` for full list)


## Contributing

Pull requests are welcome. Open an issue to suggest changes or improvements.

## Contact
 
 Peter Ugonna Obi
 
 For questions, open an issue or reach out directly.

## License

MIT License.
