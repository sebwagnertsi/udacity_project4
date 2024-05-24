import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


def clean_data():
    '''
    Cleans the data from the input file and saves it back to a new csv file.
    Creates label encoders for each categorical column and saves them as well.
    '''
    file_path = 'data/census.csv'
    df = pd.read_csv(file_path)

    df_cleaned = df.dropna()

    # Encode categorical variables
    #TODO do one hot encoding here instead!
    label_encoders = {}
    for column in df_cleaned.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_cleaned[column] = le.fit_transform(df_cleaned[column])
        label_encoders[column] = le

    for column, le in label_encoders.items():
        le_path = f'data/label_encoders/{column}_label_encoder.pkl'
        joblib.dump(le, le_path)

    print(df_cleaned.head())

    df_cleaned.to_csv('data/cleaned_census.csv', index=False)

    return df_cleaned, label_encoders

def get_clean_data():
    data, label_encoders = clean_data()
    y = data['salary']
    X = data.drop('salary', axis=1)
    
    return X, y, label_encoders

if __name__ == '__main__':
    clean_data()