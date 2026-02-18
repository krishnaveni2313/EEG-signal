from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable CORS for frontend connection
import os
from preprocess import preprocess_signal
from model import load_model, predict_seizure

app = Flask(__name__)
CORS(app)  # Allow frontend to communicate with backend

# Load the model at startup
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and return seizure prediction."""
    patient_name = request.form.get('name')
    patient_age = request.form.get('age')
    file = request.files['file']

    file_path = os.path.join("dataset", file.filename)
    file.save(file_path)

    # Preprocess the signal
    X_Res = preprocess_signal(file_path)

    # Predict seizure
    prediction = predict_seizure(model, X_Res)

    return jsonify({
        "patient_name": patient_name,
        "patient_age": patient_age,
        "prediction": prediction
    })

if __name__ == '__main__':
    app.run(debug=True)
