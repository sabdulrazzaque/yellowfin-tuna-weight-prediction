# Yellowfin Tuna Prediction Pipeline

---

## Table of Contents
1. [How It Works](#how-it-works)
2. [Benchmarks and Performance](#benchmarks-and-performance)
3. [Test Cases](#test-cases)
4. [Steps to Test](#steps-to-test)
5. [API Input/Output Example](#api-inputoutput-example)
6. [FAQs](#faqs)
7. [Known Issues/Limitations](#known-issueslimitations)
8. [Roadmap](#roadmap)
9. [Contributing](#contributing)

---

## How It Works

The pipeline consists of the following steps:

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

## Benchmarks and Performance

Performance metrics for trained models:

| Model          | R²    | MAE    | RMSE   | MAPE   |
|-----------------|-------|--------|--------|--------|
| Random Forest   | 0.92  | 3.45   | 4.56   | 12.3%  |
| XGBoost         | 0.93  | 3.20   | 4.21   | 11.8%  |
| LightGBM        | 0.91  | 3.60   | 4.70   | 12.5%  |
| CatBoost        | 0.94  | 3.10   | 4.00   | 11.5%  |

**Best performing model**: **CatBoost**

---

## Test Cases

### Example 1: Typical Gillnet Fishing Scenario
- **Input**:
  - Fishing Method: Gillnet
  - Length Frequency Species 1 (cm): 50
  - Year: 2025
  - Month: 5
  - Day: 20
  - Total Number: 500
  - Latitude: 10
  - Longitude: 75
- **Expected Result**:
  - `Log_Total_Number` auto-calculates to `6.216606`.
  - The model should predict a reasonable weight.

---

### Example 2: High Total Number with Purse Seine
- **Input**:
  - Fishing Method: Purse Seine
  - Length Frequency Species 1 (cm): 350
  - Year: 2030
  - Month: 12
  - Day: 31
  - Total Number: 10000
  - Latitude: -15
  - Longitude: 120
- **Expected Result**:
  - `Log_Total_Number` auto-calculates to `9.210440`.
  - Prediction should reflect a larger fish weight.

---

### Example 3: Minimal Values for Longline Fishing
- **Input**:
  - Fishing Method: Longline
  - Length Frequency Species 1 (cm): 0
  - Year: 2025
  - Month: 1
  - Day: 1
  - Total Number: 0
  - Latitude: 0
  - Longitude: 0
- **Expected Result**:
  - `Log_Total_Number` remains `0.000000`.
  - The model should handle zero inputs gracefully.

---

### Example 4: Edge Case - Latitude/Longitude Bounds
- **Input**:
  - Fishing Method: Gillnet
  - Length Frequency Species 1 (cm): 80
  - Year: 2024
  - Month: 7
  - Day: 10
  - Total Number: 200
  - Latitude: 90
  - Longitude: 180
- **Expected Result**:
  - `Log_Total_Number` auto-calculates to `5.303305`.
  - The system should validate that the latitude and longitude are within valid bounds and make a prediction.

---

### Example 5: Unrealistic Latitude/Longitude
- **Input**:
  - Fishing Method: Longline
  - Length Frequency Species 1 (cm): 75
  - Year: 2025
  - Month: 6
  - Day: 15
  - Total Number: 300
  - Latitude: 95
  - Longitude: 200
- **Expected Result**:
  - The server should respond with an error indicating that latitude and longitude are out of valid bounds.

---

### Example 6: Missing Inputs
- **Input**:
  - Fishing Method: Longline
  - Length Frequency Species 1 (cm): (Leave empty)
  - Year: 2025
  - Month: 5
  - Day: 20
  - Total Number: (Leave empty)
  - Latitude: 10
  - Longitude: 75
- **Expected Result**:
  - Missing inputs should default to `0.0` for numeric fields.
  - The server should still provide a prediction or error message based on the defaults.

---

## Steps to Test

1. Open the `form.html` file in a browser.
2. Enter the test inputs in the form fields.
3. Submit the form.
4. Observe the result displayed (or logged in the server logs).
5. Repeat with variations to ensure robustness.

---

## API Input/Output Example

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


## API Output Example

Here is an example of the API's JSON response:
```json
{
  "Prediction": "325.45 kg",
  "Feedback": "Log_Total_Number matches the logarithm of Total_Number."
}
```

---

## FAQs

**Q1:** Can this project be used for fish species other than Yellowfin Tuna?  
**A1:** The current model is trained specifically for Yellowfin Tuna. To adapt it for other species, you would need to retrain the model on a relevant dataset.

**Q2:** Why is the web form not working?  
**A2:** Ensure the API is running (`model_api.py`) and accessible at `http://127.0.0.1:5000`.

**Q3:** How do I deploy this project on the cloud?  
**A3:** You can use platforms like AWS, Azure, or Heroku. Start by containerizing the app using Docker.

---

## Known Issues/Limitations

- The model currently assumes the input data is well-formatted; unexpected formats may lead to errors.  
- Predictions may be less accurate for extreme outliers or missing features.  
- Web form input validation is basic and should be enhanced for production use.

---

## Roadmap

The following steps outline the planned and completed features of the project:

- [x] Implement preprocessing and data cleaning.  
- [x] Train and evaluate multiple models.  
- [x] Develop REST API for predictions.  
- [x] Add web form for user interaction.  
- [ ] Improve input validation in the API.  
- [ ] Add more robust error handling.  
- [ ] Deploy on cloud (AWS, Azure, etc.).  

---

## Contributing

Contributions are welcome! Here's how you can help improve the project:

1. **Fork the repository**:  
   - Click the "Fork" button in the top-right corner of the repository page.

2. **Create a feature branch**:  
   ```bash
   git checkout -b feature-name
   ```

3. **Commit your changes**:  
   ```bash
   git commit -m "Add new feature"
   ```

4. **Push to the branch**:  
   ```bash
   git push origin feature-name
   ```

5. **Submit a pull request for review**:  
   Open a pull request on the original repository and describe the changes you've made.

For major changes, please open an issue first to discuss what you would like to change. This helps maintainers provide feedback and ensure your contributions align with the project's goals.

---
