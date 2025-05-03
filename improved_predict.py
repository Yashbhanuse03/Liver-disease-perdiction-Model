import os
import pickle
import numpy as np
from improved_model import load_model, prepare_input_data

def predict_liver_disease(patient_data, model_path='saved_model/liver_disease_model.pkl', 
                         scaler_path='saved_model/scaler.pkl',
                         features_path='saved_model/feature_names.pkl'):
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
    features_path : str
        Path to the saved feature names
        
    Returns:
    --------
    tuple
        Prediction (0: No Disease, 1: Disease), probability, and severity
    """
    # Load model, scaler, and feature names
    model, scaler, feature_names = load_model(model_path, scaler_path, features_path)
    
    # Prepare input data
    X = prepare_input_data(patient_data, scaler, feature_names)
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    # Determine severity based on probability
    severity = 'Low Risk'
    if prediction == 1:
        if probability < 0.4:
            severity = 'Mild Risk'
        elif probability < 0.7:
            severity = 'Moderate Risk'
        else:
            severity = 'High Risk'
    
    return prediction, probability, severity

def get_user_input():
    """
    Get patient data from user input
    
    Returns:
    --------
    dict
        Dictionary containing patient information
    """
    print("\nEnter patient information:")
    
    patient_data = {}
    
    patient_data['Age'] = int(input("Age: "))
    
    gender = input("Gender (M/F): ").upper()
    patient_data['Gender'] = 'Male' if gender == 'M' else 'Female'
    
    patient_data['Total_Bilirubin'] = float(input("Total Bilirubin: "))
    patient_data['Direct_Bilirubin'] = float(input("Direct Bilirubin: "))
    patient_data['Alkaline_Phosphotase'] = int(input("Alkaline Phosphotase: "))
    patient_data['Alamine_Aminotransferase'] = int(input("Alamine Aminotransferase: "))
    patient_data['Aspartate_Aminotransferase'] = int(input("Aspartate Aminotransferase: "))
    patient_data['Total_Protiens'] = float(input("Total Proteins: "))
    patient_data['Albumin'] = float(input("Albumin: "))
    patient_data['Albumin_and_Globulin_Ratio'] = float(input("Albumin and Globulin Ratio: "))
    
    return patient_data

def main():
    """
    Main function to run the prediction interface
    """
    print("Improved Liver Disease Prediction System")
    print("=======================================")
    
    try:
        # Check if model exists
        model_path = 'saved_model/liver_disease_model.pkl'
        scaler_path = 'saved_model/scaler.pkl'
        features_path = 'saved_model/feature_names.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
            print("Model files not found. Please run improved_model.py first to train and save the model.")
            return
        
        print("Model loaded successfully.")
        
        while True:
            # Get patient data from user
            patient_data = get_user_input()
            
            # Make prediction
            prediction, probability, severity = predict_liver_disease(
                patient_data, model_path, scaler_path, features_path
            )
            
            # Display results
            print("\nPrediction Results:")
            print(f"Prediction: {'Liver Disease' if prediction == 1 else 'No Liver Disease'}")
            print(f"Probability of Liver Disease: {probability:.4f} ({probability*100:.2f}%)")
            
            if prediction == 1:
                print(f"Severity: {severity}")
            
            # Ask if user wants to continue
            if input("\nPredict for another patient? (y/n): ").lower() != 'y':
                break
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
