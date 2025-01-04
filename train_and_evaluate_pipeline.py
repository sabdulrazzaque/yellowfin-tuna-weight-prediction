import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import joblib
import logging
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = r"D:\Data_science_Fish\tuna\Tuna_5"
DATA_PATH = os.path.join(BASE_DIR, "Yellowfin_Data.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EVAL_DIR = os.path.join(RESULTS_DIR, "train_and_evaluation")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

try:
    # Load and validate data
    logger.info("Loading data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    data = pd.read_excel(DATA_PATH)

    required_columns = ['Fish', 'Total_Weight_kg', 'Fishing_Method', 'Length_Frequency_Species1_cm', 'Catch_Date', 'Total_Number']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    for col in required_columns:
        if data[col].isnull().sum() > 0:
            logger.warning(f"Column {col} contains missing values.")

    # Feature engineering
    logger.info("Performing feature engineering...")
    data['Catch_Date'] = pd.to_datetime(data['Catch_Date'])
    data['Month'] = data['Catch_Date'].dt.month
    data['Day'] = data['Catch_Date'].dt.day
    data['Log_Total_Number'] = np.log1p(data['Total_Number'])

    # Filter target variable
    data = data[data['Fish'] == 'YFT']
    if data['Total_Weight_kg'].isnull().sum() > 0:
        logger.warning("Dropping rows with NaN in Total_Weight_kg...")
        data = data.dropna(subset=['Total_Weight_kg']).copy()

    y = np.log1p(data['Total_Weight_kg'])
    categorical_features = ['Fishing_Method']
    numerical_features = [col for col in data.columns if col not in categorical_features + ['Fish', 'Total_Weight_kg', 'Catch_Date', 'Year', 'Individual_ID']]

    X = data[categorical_features + numerical_features]
    X.fillna(0, inplace=True)

    # Data Distribution Analysis
    logger.info("Analyzing data distributions...")
    for feature in numerical_features + categorical_features:
        plt.figure()
        if feature in categorical_features:
            data[feature].value_counts().plot(kind='bar')
        else:
            data[feature].hist(bins=30)
        plt.title(f"Distribution of {feature}")
        plt.savefig(os.path.join(EVAL_DIR, f"{feature}_distribution.png"))
        plt.close()

    # Preprocessing pipeline
    logger.info("Building preprocessing pipeline...")
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    preprocessor.fit(X)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "feature_pipeline.pkl"))

    # Train-validation-test split
    logger.info("Splitting data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Scale target variable
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, "scaler_y.pkl"))

    # K-Fold Cross Validation
    logger.info("Performing k-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(RandomForestRegressor(random_state=42, n_jobs=-1), preprocessor.transform(X_train), y_train_scaled, cv=kf, scoring='r2')
    logger.info(f"Cross-validation scores: {cross_val_scores}")

    # Train RandomForestRegressor
    logger.info("Training RandomForestRegressor...")
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_model.fit(preprocessor.transform(X_train), y_train_scaled)
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))

    # Algorithm Benchmarking
    models = {
        "RandomForest": rf_model,
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(random_state=42, verbose=0)
    }

    model_scores = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(preprocessor.transform(X_train), y_train_scaled)
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name.lower()}_model.pkl"))
        predictions = model.predict(preprocessor.transform(X_val))
        r2 = r2_score(y_val, scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten())
        model_scores[name] = r2
        logger.info(f"{name} Validation R²: {r2}")

    # Identify and log the best model
    best_model_name = max(model_scores, key=model_scores.get)
    best_model_score = model_scores[best_model_name]
    logger.info(f"Best Model: {best_model_name} with Validation R²: {best_model_score}")

    logger.info("Train and evaluation pipeline completed successfully.")

except Exception as e:
    logger.error(f"An error occurred: {e}")
