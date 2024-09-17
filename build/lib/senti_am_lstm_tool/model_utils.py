import os
from tensorflow.keras.models import load_model
import joblib

def load_models():
    """ Load models and scalers from the 'models' folder. """
    # Define the path to the models folder
    model_folder = os.path.join(os.path.dirname(__file__), '../models')
    
    # Load models and scalers
    am_lstm_model = load_model(os.path.join(model_folder, 'attention_model.h5'))
    scaler_features = joblib.load(os.path.join(model_folder, 'scaler_features.pkl'))
    scaler_target = joblib.load(os.path.join(model_folder, 'scaler_target.pkl'))
    
    return am_lstm_model, scaler_features, scaler_target
