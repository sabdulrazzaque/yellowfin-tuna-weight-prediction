import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = r"D:\Data_science_Fish\tuna\Tuna_5"
DATA_PATH = os.path.join(BASE_DIR, "Yellowfin_Data.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load preprocessor, models, and scaler
preprocessor_path = os.path.join(MODEL_DIR, "feature_pipeline.pkl")
scaler_y_path = os.path.join(MODEL_DIR, "scaler_y.pkl")

logger.info("Loading preprocessor and scaler...")
preprocessor = joblib.load(preprocessor_path)
scaler_y = joblib.load(scaler_y_path)

# Load trained models
model_names = ["randomforest_model.pkl", "gradientboosting_model.pkl", "lightgbm_model.pkl", "catboost_model.pkl"]
models = {}
for model_name in model_names:
    model_path = os.path.join(MODEL_DIR, model_name)
    if os.path.exists(model_path):
        model_key = model_name.replace("_model.pkl", "").capitalize()
        models[model_key] = joblib.load(model_path)

# Load and preprocess test data
logger.info("Loading test data...")
data = pd.read_excel(DATA_PATH)

# Ensure necessary columns exist
required_columns = ['Fish', 'Total_Weight_kg', 'Fishing_Method', 'Length_Frequency_Species1_cm', 'Catch_Date', 'Total_Number']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in test data: {missing_columns}")

# Data preparation
logger.info("Preprocessing test data...")
data['Catch_Date'] = pd.to_datetime(data['Catch_Date'])
data['Month'] = data['Catch_Date'].dt.month
data['Day'] = data['Catch_Date'].dt.day
data['Log_Total_Number'] = np.log1p(data['Total_Number'])
data = data[data['Fish'] == 'YFT']

# Handle missing values in target column
if data['Total_Weight_kg'].isnull().any():
    logger.warning("Target column contains NaN values. Dropping these rows.")
    data = data.dropna(subset=['Total_Weight_kg']).copy()

y_test = np.log1p(data['Total_Weight_kg'])

# Define features
categorical_features = ['Fishing_Method']
numerical_features = [col for col in data.columns if col not in categorical_features + ['Fish', 'Total_Weight_kg', 'Catch_Date']]
X_test = data[categorical_features + numerical_features].copy()

# Handle missing values in feature columns
X_test = X_test.fillna(0)

# Preprocess features and scale target
X_test_processed = preprocessor.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Evaluate all models
results = []
logger.info("Evaluating models on test data...")
for name, model in models.items():
    predictions_scaled = model.predict(X_test_processed)
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # Metrics calculation
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions)

    # Log results for each model
    logger.info(f"{name} Model Evaluation Metrics:")
    logger.info(f"R²: {r2}")
    logger.info(f"MAE: {mae}")
    logger.info(f"RMSE: {rmse}")
    logger.info(f"MAPE: {mape}")

    # Append to results
    results.append({
        "Model": name,
        "R²": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    })

# Save evaluation results
results_df = pd.DataFrame(results)
results_path = os.path.join(RESULTS_DIR, "test_results.csv")
results_df.to_csv(results_path, index=False)
logger.info(f"Test evaluation results saved at {results_path}")

# Identify and log the best model based on R²
best_model = max(results, key=lambda x: x["R²"])
logger.info(f"Best Model on Test Data: {best_model['Model']} with R²: {best_model['R²']}")