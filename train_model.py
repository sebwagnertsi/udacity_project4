import pandas as pd
import joblib

from ml.model import train_model
from ml.data import preprocess_data
from preprocessing import get_clean_training_data
from config import Config


def train_and_store_model():

    data = get_clean_training_data()
    X, y = preprocess_data(data, training=True)

    model = train_model(X, y)

    # Store the model into the data folder
    joblib.dump(model, Config.model_output_path)

    return model


if __name__ == '__main__':
    train_and_store_model()
