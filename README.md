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
git clone https://github.com/sabdulrazzaque/yellowfin-tuna-prediction.git
cd yellowfin-tuna-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Prerequisites

- **Programming Language**: Python
- **Libraries**:
  - Machine Learning: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
  - Preprocessing: `pandas`, `numpy`
  - Visualizations: `matplotlib`, `seaborn`, `shap`
  - API Development: `Flask`, `Flask-CORS`
- **Tools**:
  - Environment Management: `pip`, `venv`
  - Version Control: `Git`

---

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

### Pipeline Architecture
```plaintext
Data Preprocessing --> Model Training --> Evaluation --> Deployment --> Prediction API

## How It Works

1. **Data Preprocessing**:
   - Cleans the dataset.
   - Encodes categorical variables and scales numerical ones.

2. **Model Training**:
   - Trains multiple models using `train_and_evaluate_pipeline.py`.
   - Performs hyperparameter optimization.
   - Saves the best model.

3. **Testing**:
   - Tests the best model on unseen data using `test_best_model.py`.
   - Generates evaluation metrics.

4. **Deployment**:
   - Exposes a REST API for predictions using Flask (`model_api.py`).
   - Accepts JSON inputs and returns predictions.

5. **Web Interface**:
   - User-friendly form to input data and view predictions.
   - Communicates with the Flask API in real-time.

---

---

## Benchmarks and Performance

Include performance metrics for the trained models:
- Training/validation R², RMSE, MAE, etc.
- Compare metrics across models.

| Model          | R²    | MAE    | RMSE   | MAPE   |
|-----------------|-------|--------|--------|--------|
| Random Forest   | 0.92  | 3.45   | 4.56   | 12.3%  |
| XGBoost         | 0.93  | 3.20   | 4.21   | 11.8%  |
| LightGBM        | 0.91  | 3.60   | 4.70   | 12.5%  |
| CatBoost        | 0.94  | 3.10   | 4.00   | 11.5%  |

Best performing model: **CatBoost**


### API Input Example:
Send a JSON payload to the `/predict` endpoint:
```json
{
  "Fishing_Method": "Gillnet",
  "Length_Frequency_Species1_cm": 45.0,
  "Year": 2023,
  "Month": 10,
  "Day": 5,
  "Total_Number": 120,
  "Latitude": -15.5,
  "Longitude": 150.2
}

#API Output

{
  "Prediction": "325.45 kg",
  "Feedback": "Log_Total_Number matches the logarithm of Total_Number."
}

---

---

## FAQs

**Q1. Can this project be used for fish species other than Yellowfin Tuna?**  
A1. The current model is trained specifically for Yellowfin Tuna. To adapt it for other species, you would need to retrain the model on a relevant dataset.

**Q2. Why is the web form not working?**  
A2. Ensure the API is running (`model_api.py`) and accessible at `http://127.0.0.1:5000`.

**Q3. How do I deploy this project on the cloud?**  
A3. You can use platforms like AWS, Azure, or Heroku. Start by containerizing the app using Docker.

---

---

## Known Issues/Limitations

- The model currently assumes the input data is well-formatted; unexpected formats may lead to errors.
- Predictions may be less accurate for extreme outliers or missing features.
- Web form input validation is basic and should be enhanced for production use.

---

---

## Roadmap

- [x] Implement preprocessing and data cleaning.
- [x] Train and evaluate multiple models.
- [x] Develop REST API for predictions.
- [x] Add web form for user interaction.
- [ ] Improve input validation in the API.
- [ ] Add more robust error handling.
- [ ] Deploy on cloud (AWS, Azure, etc.).

---

---

## Contributing

Contributions are welcome! Create issues or submit pull requests to improve the project.
---

