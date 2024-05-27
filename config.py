
from sklearn.ensemble import RandomForestClassifier

class Config:
    '''
    Set all the hyperparameters and run configurations here.
    '''

    label_column = 'salary'
    
    data_path = 'data/census.csv'
    data_cleaned_path = 'data/cleaned_census.csv'
    data_preprocessed_path = 'data/'
    encoders_path = 'data/encoders'
    
    eval_mode = 'cross_val' # 'cross_val' or 'test_split
    eval_test_split = 0.3
    eval_slice_output_path = 'slice_output.txt'

    model_output_path = 'model/model.pkl'

    def get_fresh_model():
        return RandomForestClassifier()
