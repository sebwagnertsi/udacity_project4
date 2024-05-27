
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from ml.model import compute_model_metrics
from ml.data import preprocess_data
from preprocess_data import get_clean_training_data
from config import Config


def run_cross_validation():

    data = get_clean_training_data()
    X, y = preprocess_data(data, training=True) # Apply the column transformations on the data, and retrieve the y column
    model = Config.get_fresh_model()

    # Perform 10-fold cross validation
    scores = cross_val_score(model, X, y, cv=10)
    print("Cross Validation Scores:", scores)

def run_train_test_evaluation():
    data = get_clean_training_data()
    X, y = preprocess_data(data, training=True) # Apply the column transformations on the data, and retrieve the y column
    model = Config.get_fresh_model()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.eval_test_split, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print(f"Precision: {precision}, Recall: {recall}, F1: {fbeta}")
    return precision, recall, fbeta

def run_slice_evaluation(model, slice_name):

    data = get_clean_training_data()
    X, y = preprocess_data(data, training=True) # Apply the column transformations on the data, and retrieve the y column
    
    preds = model.predict(X)
    precision, recall, fbeta = compute_model_metrics(y, preds)

    print(f"Precision: {precision}, Recall: {recall}, F1: {fbeta}")
    return precision, recall, fbeta


if __name__ == '__main__':
    
    run_train_test_evaluation()