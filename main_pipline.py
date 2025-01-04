import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from argparse import ArgumentParser
import traceback

# Argument parsing for configurable options
parser = ArgumentParser(description="Run machine learning pipeline for fish weight prediction.")
parser.add_argument("--rare-threshold", type=float, default=0.05, help="Threshold for rare category replacement.")
parser.add_argument("--test-size", type=float, default=0.2, help="Test set size as a proportion.")
parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
args = parser.parse_args()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = r"D:\Data_science_Fish\tuna\Tuna_5\Yellowfin_Data.xlsx"
BASE_DIR = os.path.dirname(DATA_PATH)
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

try:
    # Load and validate data
    logger.info("Loading data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    data = pd.read_excel(DATA_PATH)

    required_columns = ['Fish', 'Total_Weight_kg', 'Fishing_Method', 'Length_Frequency_Species1_cm', 'Catch_Date']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    for col in required_columns:
        if data[col].isnull().sum() > 0:
            logger.warning(f"Column {col} contains missing values.")

    # Feature engineering
    logger.info("Performing feature engineering...")
    data.fillna(data.median(numeric_only=True), inplace=True)
    data['Catch_Date'] = pd.to_datetime(data['Catch_Date'])
    data['Year'] = data['Catch_Date'].dt.year
    data['Month'] = data['Catch_Date'].dt.month
    data['Day'] = data['Catch_Date'].dt.day
    data['Log_Total_Number'] = np.log1p(data['Total_Number'])

    rare_threshold = args.rare_threshold
    data['Fishing_Method'] = data['Fishing_Method'].replace(
        data['Fishing_Method'].value_counts(normalize=True)[
            data['Fishing_Method'].value_counts(normalize=True) < rare_threshold
        ].index, 'Other'
    )

    # Filter target variable
    data = data[data['Fish'] == 'YFT']
    y = np.log1p(data['Total_Weight_kg'])
    categorical_features = ['Fishing_Method']
    numerical_features = [col for col in data.columns if col not in categorical_features + ['Fish', 'Total_Weight_kg', 'Catch_Date']]
    X = data[categorical_features + numerical_features]

    # Preprocessing pipeline
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
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "feature_pipeline.pkl"))
    logger.info(f"Feature pipeline saved at {MODEL_DIR}")

    # Train-test split
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, "scaler_y.pkl"))
    logger.info(f"Target scaler saved at {MODEL_DIR}")

    # Function for SHAP analysis
    def shap_analysis(model, X_train, X_test, feature_names, model_name):
        try:
            # Skip SHAP for complex ensemble models like StackingRegressor
            if isinstance(model, StackingRegressor):
                logger.warning(f"SHAP analysis skipped for {model_name} due to model complexity.")
                return
            
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            shap.summary_plot(shap_values, feature_names=feature_names, show=False)
            plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_shap_summary.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Error during SHAP analysis for {model_name}: {e}")
        
    #def shap_analysis(model, X_train, X_test, feature_names, model_name):
    #    try:
     #       explainer = shap.Explainer(model, X_train)
      #      shap_values = explainer(X_test)
       #     shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        #    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_shap_summary.png"))
        #    plt.close()
       # except Exception as e:
        #    logger.error(f"Error during SHAP analysis for {model_name}: {e}")

    # Train and evaluate
    def train_and_evaluate(models, X_train, X_test, y_train, y_test, preprocessor):
        results = []
        best_model = None
        best_r2 = -float("inf")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        numerical_feature_names = np.array(preprocessor.transformers_[0][2])
        categorical_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2])
        feature_names = np.concatenate([numerical_feature_names, categorical_feature_names])

        for name, model_info in models.items():
            try:
                logger.info(f"Training {name}...")
                model = model_info['model']
                param_grid = model_info.get('param_grid', None)

                if param_grid:
                    model = RandomizedSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='r2', random_state=args.random_state)
                    model.fit(X_train_processed, y_train)
                    logger.info(f"Best parameters for {name}: {model.best_params_}")
                    model = model.best_estimator_

                model.fit(X_train_processed, y_train)
                predictions = model.predict(X_test_processed)

                metrics = {
                    "Model": name,
                    "R²": r2_score(y_test, predictions),
                    "MAE": mean_absolute_error(y_test, predictions),
                    "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                    "MAPE": mean_absolute_percentage_error(y_test, predictions),
                    "Explained Variance": r2_score(y_test, predictions)
                }
                results.append(metrics)
                logger.info(f"Metrics for {name}: {metrics}")

                plt.figure()
                sns.residplot(x=y_test, y=predictions, lowess=True)
                plt.xlabel("Actual")
                plt.ylabel("Residuals")
                plt.title(f"Residual Plot: {name}")
                plt.savefig(os.path.join(PLOTS_DIR, f"{name}_residual_plot.png"))
                plt.close()

                shap_analysis(model, X_train_processed, X_test_processed, feature_names, model_name=name)

                if metrics["R²"] > best_r2:
                    best_r2 = metrics["R²"]
                    best_model = model
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                logger.error(traceback.format_exc())

        results_df = pd.DataFrame(results)
        results_path = os.path.join(RESULTS_DIR, "model_results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Model results saved at {results_path}")

        return best_model

    # Models
    models = {
        "Random Forest": {
            "model": RandomForestRegressor(random_state=args.random_state),
            "param_grid": {"n_estimators": [100, 200], "max_depth": [10, 20, None]}
        },
        "XGBoost": {
            "model": XGBRegressor(random_state=args.random_state),
            "param_grid": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [6, 10]}
        },
        "LightGBM": {
            "model": LGBMRegressor(random_state=args.random_state),
            "param_grid": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [-1, 10]}
        },
        "CatBoost": {
            "model": CatBoostRegressor(random_state=args.random_state, verbose=0),
            "param_grid": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "depth": [6, 8]}
        },
        "Stacking": {
            "model": StackingRegressor(estimators=[
                ('rf', RandomForestRegressor(random_state=args.random_state)),
                ('xgb', XGBRegressor(random_state=args.random_state)),
                ('lgbm', LGBMRegressor(random_state=args.random_state)),
                ('cat', CatBoostRegressor(random_state=args.random_state, verbose=0))
            ])
        }
    }

    best_model = train_and_evaluate(models, X_train, X_test, y_train_scaled, y_test_scaled, preprocessor)
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "feature_pipeline.pkl"))
    logger.info(f"Fitted feature pipeline saved at {MODEL_DIR}")
    #joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    #logger.info(f"Best model saved at {MODEL_DIR}")

except Exception as e:
    logger.error(f"An error occurred: {e}")
    logger.error(traceback.format_exc())
