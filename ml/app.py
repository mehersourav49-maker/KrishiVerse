# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
# Load trained model + label encoder
model_data = joblib.load("model.joblib")
model = model_data['model']
label_encoder = model_data['label_encoder']

app = Flask(__name__)
CORS(app)  # Allow frontend (HTML/JS) to call API

@app.route('/')
def home():
    return "🌱 Crop Recommendation API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        N = data.get('N')
        P = data.get('P')
        K = data.get('K')
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        ph = data.get('ph')
        rainfall = data.get('rainfall')

        # Ensure all values are provided
        if None in [N, P, K, temperature, humidity, ph, rainfall]:
            return jsonify({"error": "Missing input values"}), 400

        # Prepare input for model
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        crop_name = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"recommended_crop": crop_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
