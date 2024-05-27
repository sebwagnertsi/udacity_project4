from train_model import train_and_store_model, run_cross_validation, run_train_test_evaluation
from preprocess_data import clean_data, get_clean_training_data
from evaluation import run_all_slices_evaluation



def run_full_process():
    print("Running full process")

    # Preprocess the data
    print("Cleaning and preprocessing data.")
    X, y = clean_data()

    # Train the model
    print("Evaluating model performance.")
    precision, recall, fbeta = run_train_test_evaluation()
    
    # Training final model
    model = train_and_store_model()

    # Run the all slices evaluation with model
    run_all_slices_evaluation(model)

if __name__=='__main__':
    run_full_process()
    