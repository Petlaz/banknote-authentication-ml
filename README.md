# Banknote Authentication ML

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A machine learning project to classify banknotes as genuine or forged using statistical features extracted from images.

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

## Requirements
- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- (see `requirements.txt` for full list)

## Model Details
- **Algorithm:** Voting Classifier (ensemble of Random Forest, XGBoost), SVM, KNN
- **Features Used:** Variance, Skewness, Kurtosis, Entropy of wavelet-transformed images
- **Scaling:** StandardScaler
- **Test Size:** 20%

Example Usage
Predict a new banknote in Python:
python
import joblib
import numpy as np

model = joblib.load("models/best_model.pkl")

Example features: [variance, skewness, kurtosis, entropy]

sample = np.array([[2.3, 6.7, -1.2, 0.5]])

* Scale the sample

scaled_sample = scaler.transform(new_sample)

predict class
prediction = model.predict(scaled_sample)

## Output result

print("Prediction:", "Authentic" if prediction[0] == 0 else "Forged")

 
## Insights

- The model achieves perfect accuracy on the test set, indicating strong separability in the data.
- Feature scaling and ensemble methods contributed to robust performance.
- The confusion matrix shows no misclassifications.


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
               

## Visualizations

Here’s the confusion matrix from the final model:

![Confusion Matrix](reports/figures/confusion_matrix.png)

Accuracy and classification report are saved in:
- [`reports/metrics/classification_report.txt`](reports/metrics/classification_report.txt)

- [`reports/metrics/classification_report.json`](reports/metrics/classification_report.json)


## Quickstart

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the model**
   ```bash
   python -m banknote_auth.modeling.train
   ```
4. **Prediction**
   '''bash
   python -m banknote_auth.modeling.prediction
   '''
5. **Generate evaluation report**
   ```bash
   python -m banknote_auth.reporting.generate_report
   ``'

## Deployment

You can deploy the interactive apps using either Gradio or Streamlit.

1. **Gradio App**

python app_gradio.py

This will launch a local Gradio web interface for banknote authentication.

2. **Streamlit App**

streamlit run app_streamlit.py

This will launch a local Streamlit dashboard for banknote authentication.


## Apps

1. **Gradio App**

File: app_gradio.py

Launches a simple web interface for uploading features and getting predictions.

Example screenshot:

Gradio App Screenshot <!-- Add your screenshot if available -->

2. **Streamlit App**

File: app_streamlit.py

Provides an interactive dashboard for exploring predictions and model metrics.

Example screenshot:

Streamlit App Screenshot <!-- Add your screenshot if available -->


## Online Demo 

You can deploy these apps to Hugging Face Spaces (for Gradio) or Streamlit Community Cloud for free!


## Data Source

- [UCI ML Repository: Banknote Authentication Data Set](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)


## Results

The VotingClassifier achieved the best overall results:

* F1 Score: 0.994

* ROC AUC: 1.00

* Recall: 1.00 (No false negatives)

This ensemble model is robust, interpretable, and ready for deployment in real-world banknote authentication systems.

## FAQ

**Q: How do I add new data?**  
A: Place new CSV files in `data/raw/` and update your data processing scripts.

**Q: Can I use a different classifier?**  
A: Yes! Modify `banknote_auth/modeling/models.py` to build and train other classifiers.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.


## Contact

For questions, open an issue or contact [Peter Ugonna Obi](email:peter.obi96@yahoo.com).

## License

This project is licensed under the MIT License.

