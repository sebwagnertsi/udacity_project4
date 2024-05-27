import pandas as pd
from numpy import ndarray
from config import Config
from ml.data import preprocess_data

def clean_data():
    '''
    Cleans the data from the input file and saves it back to a new csv file.
    '''
    
    df = pd.read_csv(Config.data_path)
    df_cleaned = df.dropna()
    df_cleaned.to_csv(Config.data_cleaned_path, index=False)

    # Now initialize the encoders
    X, y = preprocess_data(df_cleaned, initialize=True)

    # Save all the X and y values:
    pd.DataFrame(X).to_csv(Config.data_preprocessed_path+'/_X.csv', index=False)
    pd.DataFrame(y).to_csv(Config.data_preprocessed_path+'/_y.csv', index=False)

    return X, y

def get_clean_training_data():
    try:
        data = pd.read_csv(Config.data_cleaned_path)
    except FileExistsError:
        data = clean_data()

    return data
    
if __name__ == '__main__':
    clean_data()
