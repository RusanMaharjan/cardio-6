import joblib

MODEL_PATH = 'models/logistic/logistic_model.pkl'
SCALER_PATH = 'models/logistic/logistic_scaler.pkl'

def load_logistic_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    return model, scaler