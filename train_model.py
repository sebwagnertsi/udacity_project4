import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from ml.model import train_model, compute_model_metrics
from preprocess_data import get_clean_data

def train_and_store_model():

    X, y, label_encoders = get_clean_data()    
    model = train_model(X, y)

    # Store the model into the data folder
    joblib.dump(model, 'model/model.pkl')


def run_cross_validation():

    X, y, label_encoders = get_clean_data()

    model = RandomForestClassifier()

    # Perform 10-fold cross validation
    scores = cross_val_score(model, X, y, cv=10)
    print("Cross Validation Scores:", scores)
    


if __name__ == '__main__':
    
    run_cross_validation()