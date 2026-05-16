# Student Dropout Prediction - COMP 30044 CW2

This project predicts whether a student is likely to Dropout or Graduate using the UCI Predict Students' Dropout and Academic Success dataset.

## Dataset Source

UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

## Task Type

Binary classification: Dropout vs Graduate

## Models Used

- Logistic Regression
- Basic Random Forest
- Tuned Random Forest using GridSearchCV with recall scoring
- Gradient Boosting

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

## Key Output Files

- model_results.csv
- report_values.txt
- student_dropout_model.pkl
- scaler.pkl
- target_encoder.pkl
- feature_columns.pkl
- figures/

## Setup Instructions

```bash
pip install -r requirements.txt
```

## Run the Main Machine Learning Script

```bash
python3 student_dropout_prediction.py
```

## Run the Streamlit App

```bash
streamlit run app.py
```

## Ethical Note

This project is for educational demonstration only and should not be used as the only basis for real academic decisions.
