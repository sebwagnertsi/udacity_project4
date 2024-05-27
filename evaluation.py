
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

def run_slice_evaluation(model, data, slice_name):

    X, y = preprocess_data(data, training=True) # Apply the column transformations on the data, and retrieve the y column
    
    slice_preds = model.predict(X)
    precision, recall, fbeta = compute_model_metrics(y, slice_preds)

    distinct_values = data[slice_name].unique()

    output = ''
    results = {}
    for dv in distinct_values:
        indices = data[data[slice_name] == dv].index
        # print(f"Indices where {slice_name} equals {dv}: {indices}")
        y_slice = y[indices]
        X_slice = X[indices]

        slice_preds = model.predict(X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, slice_preds)

        output = f"Column: {slice_name} | Value: {dv} | Precision: {precision}, Recall: {recall}, F1: {fbeta}\n"
        print(output)

        key = f'{slice_name}___{dv}'
        results[key] = {"precision": precision, "recall": recall, "f1": fbeta}
    return results, output

def run_all_slices_evaluation():
    with open(Config.model_output_path, 'rb') as f:
        model = joblib.load(f)
    
        data = get_clean_training_data()

        categorical_features = set()
        for column in data.select_dtypes(include=['object']).columns:
            categorical_features.add(column)        
        categorical_features.remove(Config.label_column)

        results = []
        output_text = ''
        for category in categorical_features:
            res, output = run_slice_evaluation(model, data, category)
            results.append(res)
            output_text += output

        with open(Config.eval_slice_output_path, 'w') as f:
            joblib.dump(output, f)


if __name__ == '__main__':
    run_all_slices_evaluation()
    