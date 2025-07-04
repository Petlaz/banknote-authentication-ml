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

## Quictstarts

# 1. Clone the repository
# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python -m banknote_auth.modeling.train

# 4. Generate report (accuracy, confusion matrix, etc.)
python -m banknote_auth.reporting.generate_report

# 5. Run prediction
python -m banknote_auth.modeling.predict


## Model Details

* **Algorithm: VotingClassifier (ensemble of Random Forest, XGBoost, SVM, KNN)**

* **Features Used: Variance, Skewness, Kurtosis, Entropy**

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

- **Gradio App**

python app_gradio.py

Launches a local Gradio web interface.

- **Streamlit App**

streamlit run app_streamlit.py

Launches a local Streamlit dashboard.

- **Screenshot Demo**

Add screenshots to reports/figures/ and reference them in README.md or docs/index.md


## Online Demo

You can deploy this project on:

* **Hugging Face Spaces (for Gradio)**

* **Streamlit Community Cloud**


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

1. **UCI Banknote Dataset: https://archive.ics.uci.edu/ml/datasets/banknote+authentication**

2. **Hands-On ML with Scikit-Learn (A. Géron, O'Reilly)**

3. **Machine Learning Specialization (A. Ng, Coursera)**

4. **Abid, A., et al. (2021) Gradio: Hassle-Free Sharing and Testing of ML Models
https://gradio.app**

5. **Streamlit Inc. (2021). Streamlit: The fastest way to build data apps
https://streamlit.io**



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

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.


## Contact
 
 Peter Ugonna Obi – for questions, open an issue or reach out directly.


## License

MIT License.

