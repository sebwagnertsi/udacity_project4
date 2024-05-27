import pandas as pd
import joblib

from ml.model import train_model
from evaluation import run_train_test_evaluation
from preprocess_data import get_clean_training_data
from config import Config

def train_and_store_model():

    X, y = get_clean_training_data()    
    model = train_model(X, y)

    # Store the model into the data folder
    joblib.dump(model, Config.model_output_path)

    return model



if __name__ == '__main__':    
    train_and_store_model()