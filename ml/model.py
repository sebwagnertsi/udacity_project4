from sklearn.metrics import fbeta_score, precision_score, recall_score
from config import Config
import joblib
from pandas import DataFrame
from preprocess_data import preprocess_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Train a Classifier as specified in the config
    model = Config.get_fresh_model()
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
        
    preds = model.predict(X)
    return preds

def inference_from_df(df: DataFrame):
    '''
    Loads the model and preprocess the data to make predictions.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe with the data to make predictions on.

    Returns:
    -----
    preds: np.array
        Predictions from the model.
    '''
    with open(Config.model_output_path, 'rb') as f:
        model = joblib.load(f)
        
        X, y = preprocess_data(df, training=False)
        preds = inference(model, X)
        return preds
    return None

def convert_inf_results_to_label(preds):
    '''
    Converts the predictions from the model to the original labels.
    Inputs
    ------
    preds: np.array
        numpy array with the predictions

    Returns:
    -----
    np.array
        Predictions converted to the original labels.
    '''
    lb = joblib.load(Config.encoders_path+'/lb.pkl')
    
    return lb.inverse_transform(preds)