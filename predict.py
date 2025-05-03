import os
import pickle
from preprocessing import prepare_input_data

def load_model(model_path='saved_model/liver_disease_model.pkl',
              scaler_path='saved_model/scaler.pkl'):
    """
    Load the model and scaler from disk

    Parameters:
    -----------
    model_path : str
        Path to the saved model
    scaler_path : str
        Path to the saved scaler

    Returns:
    --------
    tuple
        Loaded model and scaler
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler not found. Please train the model first by running main.py.")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler

def predict_liver_disease(patient_data, model, scaler):
    """
    Predict liver disease for a new patient

    Parameters:
    -----------
    patient_data : dict
        Dictionary containing patient information
    model : object
        Trained model
    scaler : object
        Fitted scaler

    Returns:
    --------
    tuple
        Prediction (0: No Disease, 1: Disease) and probability
    """
    # Prepare input data
    X = prepare_input_data(patient_data, scaler)

    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return prediction, probability

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
    print("Liver Disease Prediction System")
    print("===============================")

    try:
        # Load model and scaler
        model, scaler = load_model()
        print("Model loaded successfully.")

        while True:
            # Get patient data from user
            patient_data = get_user_input()

            # Make prediction
            prediction, probability = predict_liver_disease(patient_data, model, scaler)

            # Display results
            print("\nPrediction Results:")
            print(f"Prediction: {'Liver Disease' if prediction == 1 else 'No Liver Disease'}")
            print(f"Probability of Liver Disease: {probability:.4f}")

            # Ask if user wants to continue
            if input("\nPredict for another patient? (y/n): ").lower() != 'y':
                break

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run main.py first to train and save the model.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
