import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_data():
    file_path = 'data/census.csv'
    df = pd.read_csv(file_path)

    # Drop any rows with missing values
    df_cleaned = df.dropna()

    # Encode categorical variables
    label_encoders = {}
    for column in df_cleaned.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_cleaned[column] = le.fit_transform(df_cleaned[column])
        label_encoders[column] = le

    # Display the first few rows of the cleaned and encoded dataframe
    print(df_cleaned.head())

    # If you need to save the cleaned dataframe to a new CSV file
    df_cleaned.to_csv('data/cleaned_census.csv', index=False)

if __name__ == '__main__':
    clean_data()