import os

from preprocessing import load_data, preprocess_data, prepare_input_data
from model import (train_models, evaluate_models, get_best_model, save_model,
                  load_model, print_confusion_matrix, print_feature_importance)
from sklearn.metrics import classification_report

def train_liver_disease_model(data_path='liver.csv', test_size=0.2, random_state=42):
    """
    Train a liver disease prediction model

    Parameters:
    -----------
    data_path : str
        Path to the dataset
    test_size : float
        Size of the test set
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    tuple
        Best model, scaler, and feature names
    """
    print("Loading data...")
    data = load_data(data_path)

    print("\nExploring data...")
    print(f"Dataset Shape: {data.shape}")
    print(f"\nFirst 5 rows:\n{data.head()}")
    print(f"\nMissing Values:\n{data.isnull().sum()}")
    print(f"\nClass Distribution:\n{data['Dataset'].value_counts()}")

    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data, test_size, random_state)

    print("\nTraining models...")
    models = train_models(X_train, y_train)

    print("\nEvaluating models...")
    results = evaluate_models(models, X_test, y_test)

    print("\nModel Performance:")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")

    print("\nSelecting best model...")
    best_model_name, best_model, best_metrics = get_best_model(models, results)
    print(f"Best model: {best_model_name} (F1 Score: {best_metrics['f1']:.4f})")

    print("\nDetailed evaluation of best model:")
    print(classification_report(y_test, best_metrics['y_pred'], target_names=['No Disease', 'Disease']))

    # Print confusion matrix for best model
    print_confusion_matrix(y_test, best_metrics['y_pred'],
                         title=f'Confusion Matrix - {best_model_name}')

    # Print feature importance if applicable
    feature_names = data.drop('Dataset', axis=1).columns
    print_feature_importance(best_model, feature_names)

    # Save the best model
    print("\nSaving best model...")
    model_path, scaler_path = save_model(best_model, scaler)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    return best_model, scaler, feature_names

def predict_liver_disease(patient_data, model_path='saved_model/liver_disease_model.pkl',
                         scaler_path='saved_model/scaler.pkl'):
    """
    Predict liver disease for a new patient

    Parameters:
    -----------
    patient_data : dict
        Dictionary containing patient information
    model_path : str
        Path to the saved model
    scaler_path : str
        Path to the saved scaler

    Returns:
    --------
    tuple
        Prediction (0: No Disease, 1: Disease) and probability
    """
    # Load model and scaler
    model, scaler = load_model(model_path, scaler_path)

    # Prepare input data
    X = prepare_input_data(patient_data, scaler)

    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return prediction, probability

def main():
    """
    Main function to run the liver disease prediction system
    """
    print("Liver Disease Prediction System")
    print("===============================")

    # Check if model already exists
    model_path = 'saved_model/liver_disease_model.pkl'
    scaler_path = 'saved_model/scaler.pkl'

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Found existing model. Loading...")
        model, scaler = load_model(model_path, scaler_path)
        print("Model loaded successfully.")
    else:
        print("No existing model found. Training new model...")
        model, scaler, _ = train_liver_disease_model()

    # Example prediction
    print("\nExample Prediction:")
    example_patient = {
        'Age': 45,
        'Gender': 'Male',
        'Total_Bilirubin': 1.2,
        'Direct_Bilirubin': 0.4,
        'Alkaline_Phosphotase': 290,
        'Alamine_Aminotransferase': 80,
        'Aspartate_Aminotransferase': 70,
        'Total_Protiens': 6.8,
        'Albumin': 3.3,
        'Albumin_and_Globulin_Ratio': 0.9
    }

    # Make prediction
    X = prepare_input_data(example_patient, scaler)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    print(f"Patient Data: {example_patient}")
    print(f"Prediction: {'Liver Disease' if prediction == 1 else 'No Liver Disease'}")
    print(f"Probability of Liver Disease: {probability:.4f}")

if __name__ == "__main__":
    main()
