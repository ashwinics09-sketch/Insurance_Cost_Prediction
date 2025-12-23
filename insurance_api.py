from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained ML pipeline
model = joblib.load("insurance_premium_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Insurance Premium Prediction API is running"

# ===================================
# Prediction Endpoint
# ===================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        input_data = request.get_json()

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)

        return jsonify({
            "predicted_premium": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
