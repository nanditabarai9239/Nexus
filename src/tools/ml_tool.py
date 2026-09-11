import pandas as pd
import joblib
import os
import json
import numpy as np
from langchain.tools import tool

# Global variables to cache model loading
_MODEL = None
_SCALER = None
_ENCODERS = None
_FEATURES = None
_NUM_COLS = None
_CAT_COLS = None
_TARGET_COL = None
_NUM_IMPUTER = None
_CAT_IMPUTER = None

def load_artifacts():
    global _MODEL, _SCALER, _ENCODERS, _FEATURES, _NUM_COLS, _CAT_COLS, _TARGET_COL, _NUM_IMPUTER, _CAT_IMPUTER
    if _MODEL is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(base_dir, 'models')
        _MODEL = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
        _SCALER = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        _ENCODERS = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
        _FEATURES = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
        _NUM_COLS = joblib.load(os.path.join(models_dir, 'numerical_cols.pkl'))
        _CAT_COLS = joblib.load(os.path.join(models_dir, 'categorical_cols.pkl'))
        _TARGET_COL = joblib.load(os.path.join(models_dir, 'target_col.pkl'))
        try:
            _NUM_IMPUTER = joblib.load(os.path.join(models_dir, 'num_imputer.pkl'))
            _CAT_IMPUTER = joblib.load(os.path.join(models_dir, 'cat_imputer.pkl'))
        except:
            _NUM_IMPUTER = None
            _CAT_IMPUTER = None

def make_prediction_dict(input_dict: dict) -> dict:
    """Helper to predict from a python dict. Used by API as well."""
    load_artifacts()
    
    # Create DataFrame for input
    input_data = pd.DataFrame([input_dict])
    
    # Ensure order matches features
    missing_cols = set(_FEATURES) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = np.nan # Use NaN so imputer handles it
        
    input_data = input_data[_FEATURES]
    
    # Apply imputation
    if _NUM_IMPUTER is not None and len(_NUM_COLS) > 0:
        input_data[_NUM_COLS] = _NUM_IMPUTER.transform(input_data[_NUM_COLS])
        
    if _CAT_IMPUTER is not None and len(_CAT_COLS) > 0:
        input_data[_CAT_COLS] = _CAT_IMPUTER.transform(input_data[_CAT_COLS])
    
    # Apply encoding
    for col in _CAT_COLS:
        if col in input_data.columns:
            le = _ENCODERS[col]
            val = str(input_data[col].iloc[0])
            if val in le.classes_:
                input_data[col] = le.transform([val])
            else:
                input_data[col] = 0
                
    # Apply scaling
    if len(_NUM_COLS) > 0:
        input_data[_NUM_COLS] = _SCALER.transform(input_data[_NUM_COLS])
    
    # Predict
    prediction_idx = _MODEL.predict(input_data)[0]
    prediction_prob = float(_MODEL.predict_proba(input_data)[0].max())
    
    prediction_label = _ENCODERS['__TARGET__'].inverse_transform([prediction_idx])[0]
    
    return {
        "prediction": prediction_label,
        "confidence": prediction_prob,
        "target_col": _TARGET_COL
    }

@tool
def run_inference(customer_data_json: str) -> str:
    """
    Runs an inference prediction based on a profile of features.
    Pass a JSON string with the exact feature keys and values.
    Returns the prediction and the confidence score.
    """
    try:
        data = json.loads(customer_data_json)
        result = make_prediction_dict(data)
        return f"Prediction for {result['target_col']}: {result['prediction']} (Confidence: {result['confidence']:.2%})"
    except Exception as e:
        return f"Error making prediction: {str(e)}"
