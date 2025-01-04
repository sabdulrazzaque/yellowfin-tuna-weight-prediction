# yellowfin-tuna-weight-prediction
Machine learning pipeline for predicting Yellowfin Tuna weight, including data preprocessing, model training, testing, deployment, and an interactive web-based prediction interface.


# Yellowfin Tuna Weight Prediction Pipeline

This repository provides a complete pipeline for predicting the weight of Yellowfin Tuna (YFT) from a dataset. The pipeline includes data preprocessing, model training, testing, deployment, and a web-based interface for predictions.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Pipeline Details](#pipeline-details)
6. [Directory Structure](#directory-structure)
7. [Contributing](#contributing)

---

## Project Overview

This project predicts the weight of Yellowfin Tuna using machine learning models. The pipeline consists of:
- Preprocessing data.
- Training and evaluating models.
- Testing the best model.
- Deploying the model via a REST API.
- Interacting with a web interface for predictions.

---

## Features
- **Data Preprocessing**:
  - Handles missing values, scaling, and encoding.
- **Model Training**:
  - Random Forest, XGBoost, LightGBM, CatBoost, and Stacking models.
  - Hyperparameter optimization using RandomizedSearchCV.
- **Model Evaluation**:
  - Metrics: R², MAE, RMSE, MAPE.
  - SHAP analysis for feature importance.
- **Deployment**:
  - REST API built with Flask.
- **Web Form**:
  - Interactive web interface for input and predictions.

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/yellowfin-tuna-prediction.git
cd yellowfin-tuna-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Install Additional Libraries
```bash
pip install lightgbm xgboost catboost flask flask-cors
```

---

## Usage

### Step 1: Data Preprocessing
Run the preprocessing pipeline:
```bash
python main_pipeline.py
```

### Step 2: Model Training and Evaluation
Train and evaluate models:
```bash
python train_and_evaluate_pipeline.py
```

### Step 3: Model Testing
Test the trained model:
```bash
python test_best_model.py
```

### Step 4: Start the API
Deploy the model as a REST API:
```bash
python model_api.py
```

### Step 5: Web Interface
Open `form.html` in your browser or access the app at `http://127.0.0.1:5000`.

---

## Pipeline Details

### 1. Data Preprocessing
- Script: `main_pipeline.py`.
- Tasks:
  - Load and clean the dataset.
  - Feature engineering and scaling.
  - Save the preprocessor pipeline.

### 2. Model Training and Evaluation
- Script: `train_and_evaluate_pipeline.py`.
- Tasks:
  - Train models using multiple algorithms.
  - Hyperparameter optimization.
  - Save the best-performing model.

### 3. Model Testing
- Script: `test_best_model.py`.
- Tasks:
  - Load the saved model.
  - Evaluate on unseen test data.
  - Save test results.

### 4. Model Deployment
- Script: `model_api.py`.
- Tasks:
  - Expose a REST API for predictions.
  - Validate inputs and return predictions.

### 5. Web Interface
- File: `form.html`.
- Tasks:
  - Collect user input.
  - Display predictions and feedback.

---

## Directory Structure

```plaintext
yellowfin-tuna-prediction/
├── data/
│   └── Yellowfin_Data.xlsx       # Input dataset
├── models/
│   ├── feature_pipeline.pkl     # Preprocessor pipeline
│   ├── scaler_y.pkl             # Scaler for the target variable
│   └── best_model.pkl           # Trained model
├── results/
│   ├── train_and_evaluation/    # Evaluation plots and results
│   └── test_results.csv         # Test results
├── main_pipeline.py             # Data preprocessing script
├── train_and_evaluate_pipeline.py # Model training script
├── test_best_model.py           # Model testing script
├── model_api.py                 # API for predictions
├── form.html                    # Web interface for predictions
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```

---


## Contributing

Contributions are welcome! Create issues or submit pull requests to improve the project.

