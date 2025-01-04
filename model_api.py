import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = r"D:\Data_science_Fish\tuna\Tuna_5"
MODEL_DIR = os.path.join(BASE_DIR, "models")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "feature_pipeline.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_y.pkl")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "catboost_model.pkl")

# Load model and preprocessors
try:
    logger.info("Loading model and preprocessors...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    scaler_y = joblib.load(SCALER_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    logger.info("Model and preprocessors loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or preprocessors: {e}")
    raise e


@app.route("/", methods=["GET"])
def home():
    """Home route to display the form."""
    return render_template("form.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle predictions with enhanced feedback and validation."""
    try:
        if not request.is_json:
            return jsonify({"error": "Unsupported Media Type: Content-Type must be application/json"}), 415

        data = request.json

        # Define default values
        default_values = {
            "Fishing_Method": "Gillnet",
            "Length_Frequency_Species1_cm": 0.0,
            "Year": 2025,
            "Month": 1,
            "Day": 1,
            "Log_Total_Number": 0.0,
            "Individual_ID": 0.0,
            "Total_Number": 0.0,
            "Length_Frequency_Species2_cm": 0.0,
            "Length_Frequency_Species3_cm": 0.0,
            "Weight_per_Individual_kg": 0.0,
            "Avg_Length_Species1_cm": 0.0,
            "Avg_Length_Species2_cm": 0.0,
            "Avg_Length_Species3_cm": 0.0,
            "Latitude": 0.0,
            "Longitude": 0.0,
        }

        # Parse and validate input
        input_data = {}
        for key, default in default_values.items():
            value = data.get(key, default)
            try:
                if key == "Fishing_Method":
                    input_data[key] = value if value else default
                else:
                    input_data[key] = float(value) if value else default
            except ValueError:
                return jsonify({"error": f"Invalid value for {key}. Must be a number."}), 400

        # Cross-field validation
        if input_data["Total_Number"] > 0:
            recalculated_log_total = np.log1p(input_data["Total_Number"])
            if not np.isclose(input_data["Log_Total_Number"], recalculated_log_total):
                input_data["Log_Total_Number"] = recalculated_log_total
                user_feedback = "Log_Total_Number was recalculated based on Total_Number."
            else:
                user_feedback = "Log_Total_Number matches the logarithm of Total_Number."
        else:
            user_feedback = "Total_Number is zero, no recalculation needed."

        # Check for unrealistic values
        if input_data["Latitude"] < -90 or input_data["Latitude"] > 90:
            return jsonify({"error": "Latitude should be between -90 and 90."}), 400
        if input_data["Longitude"] < -180 or input_data["Longitude"] > 180:
            return jsonify({"error": "Longitude should be between -180 and 180."}), 400

        # Preprocess input
        df_input = pd.DataFrame([input_data])
        processed_input = preprocessor.transform(df_input)

        # Make prediction
        prediction_scaled = model.predict(processed_input)
        prediction = np.expm1(scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0])

        # Return response
        return jsonify({"Prediction": f"{prediction:.2f} kg", "Feedback": user_feedback})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)