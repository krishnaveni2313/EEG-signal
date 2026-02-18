import xgboost as xgb
import numpy as np

# Load pre-trained model
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("saved_model/xgboost_model.json")  # Make sure this file exists
    return model

def predict_seizure(model, X_Res):
    """Predict whether the patient has a seizure or not."""
    y_predict = model.predict(X_Res.reshape(1, -1))
    return "Seizure" if y_predict == 0 else "Seizure-free"
