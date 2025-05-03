from flask import Flask, render_template, request, redirect, url_for
import os
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

def load_model(model_path='saved_model/liver_disease_model.pkl',
              scaler_path='saved_model/scaler.pkl',
              features_path='saved_model/feature_names.pkl'):
    """
    Load the model, scaler, and feature names from disk

    Parameters:
    -----------
    model_path : str
        Path to the saved model
    scaler_path : str
        Path to the saved scaler
    features_path : str
        Path to the saved feature names

    Returns:
    --------
    tuple
        Loaded model, scaler, and feature names
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
        raise FileNotFoundError(f"Model files not found. Please train the model first by running train_with_new_data.py.")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names

def prepare_input_data(patient_data, scaler, feature_names):
    """
    Prepare input data for prediction with feature engineering

    Parameters:
    -----------
    patient_data : dict
        Dictionary containing patient information
    scaler : StandardScaler
        Scaler used to scale the training data
    feature_names : list
        List of feature names used during training

    Returns:
    --------
    np.array
        Scaled input data ready for prediction
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])

        # Convert Gender to numerical if it's not already
        if 'Gender' in df.columns and df['Gender'].dtype == 'object':
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

        # Create engineered features
        if all(col in df.columns for col in ['Total_Bilirubin', 'Direct_Bilirubin']):
            df['Bilirubin_Ratio'] = np.where(
                df['Total_Bilirubin'] > 0,
                df['Direct_Bilirubin'] / df['Total_Bilirubin'],
                0
            )

        if all(col in df.columns for col in ['Aspartate_Aminotransferase', 'Alamine_Aminotransferase']):
            df['AST_ALT_Ratio'] = np.where(
                df['Alamine_Aminotransferase'] > 0,
                df['Aspartate_Aminotransferase'] / df['Alamine_Aminotransferase'],
                0
            )

        # Add disease type columns (one-hot encoded)
        disease_types = ['Liver_Disease_Type_Fatty Liver', 'Liver_Disease_Type_Fibrosis',
                        'Liver_Disease_Type_Hepatitis', 'Liver_Disease_Type_Liver Cancer']

        for col in disease_types:
            if col not in df.columns and col in feature_names:
                df[col] = 0  # Default to 0 for all disease types

        # Handle any NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # Create a new DataFrame with all required features
        input_df = pd.DataFrame(columns=feature_names)

        # Fill in the values from our original DataFrame
        for feature in feature_names:
            if feature in df.columns:
                input_df[feature] = df[feature]
            else:
                input_df[feature] = 0

        # Scale the data
        scaled_data = scaler.transform(input_df)

        return scaled_data
    except Exception as e:
        print(f"Error in prepare_input_data: {str(e)}")
        raise

def predict_liver_disease(patient_data, model, scaler, feature_names):
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
    feature_names : list
        List of feature names used during training

    Returns:
    --------
    tuple
        Prediction (0: No Disease, 1: Disease), probability, and severity
    """
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Validate form data
            required_fields = ['age', 'gender', 'total_bilirubin', 'direct_bilirubin',
                             'alkaline_phosphotase', 'alamine_aminotransferase',
                             'aspartate_aminotransferase', 'total_proteins',
                             'albumin', 'albumin_globulin_ratio']

            for field in required_fields:
                if field not in request.form or not request.form[field]:
                    return render_template('index.html',
                                          error=f"Please fill in all required fields. Missing: {field}")

            # Get form data
            try:
                patient_data = {
                    'Age': int(request.form['age']),
                    'Gender': request.form['gender'],
                    'Total_Bilirubin': float(request.form['total_bilirubin']),
                    'Direct_Bilirubin': float(request.form['direct_bilirubin']),
                    'Alkaline_Phosphotase': int(request.form['alkaline_phosphotase']),
                    'Alamine_Aminotransferase': int(request.form['alamine_aminotransferase']),
                    'Aspartate_Aminotransferase': int(request.form['aspartate_aminotransferase']),
                    'Total_Protiens': float(request.form['total_proteins']),
                    'Albumin': float(request.form['albumin']),
                    'Albumin_and_Globulin_Ratio': float(request.form['albumin_globulin_ratio'])
                }
            except ValueError as e:
                return render_template('index.html',
                                     error=f"Invalid input: Please enter valid numeric values for all fields.")

            # Load model and scaler
            try:
                model, scaler, feature_names = load_model()
            except FileNotFoundError as e:
                return render_template('index.html',
                                     error=f"Model not found: {str(e)}")

            # Make prediction
            try:
                prediction, probability, severity = predict_liver_disease(patient_data, model, scaler, feature_names)
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                return render_template('index.html',
                                     error=f"Error making prediction: {str(e)}")

            # Calculate risk factors
            risk_factors = []

            # Check for elevated liver enzymes
            if patient_data['Alamine_Aminotransferase'] > 55:
                risk_factors.append(f"Elevated ALT: {patient_data['Alamine_Aminotransferase']} IU/L (Normal: 7-55 IU/L)")

            if patient_data['Aspartate_Aminotransferase'] > 48:
                risk_factors.append(f"Elevated AST: {patient_data['Aspartate_Aminotransferase']} IU/L (Normal: 8-48 IU/L)")

            if patient_data['Alkaline_Phosphotase'] > 147:
                risk_factors.append(f"Elevated ALP: {patient_data['Alkaline_Phosphotase']} IU/L (Normal: 44-147 IU/L)")

            # Check for elevated bilirubin
            if patient_data['Total_Bilirubin'] > 1.2:
                risk_factors.append(f"Elevated Total Bilirubin: {patient_data['Total_Bilirubin']} mg/dL (Normal: 0.1-1.2 mg/dL)")

            if patient_data['Direct_Bilirubin'] > 0.3:
                risk_factors.append(f"Elevated Direct Bilirubin: {patient_data['Direct_Bilirubin']} mg/dL (Normal: 0-0.3 mg/dL)")

            # Check for abnormal protein levels
            if patient_data['Albumin'] < 3.5:
                risk_factors.append(f"Low Albumin: {patient_data['Albumin']} g/dL (Normal: 3.5-5.0 g/dL)")

            if patient_data['Albumin_and_Globulin_Ratio'] < 1.0:
                risk_factors.append(f"Low A/G Ratio: {patient_data['Albumin_and_Globulin_Ratio']} (Normal: 1.0-2.5)")

            # Prepare result
            result = {
                'prediction': 'Liver Disease' if prediction == 1 else 'No Liver Disease',
                'probability': round(probability * 100, 2),
                'severity': severity,
                'risk_factors': risk_factors,
                'patient_data': patient_data
            }

            return render_template('result.html', result=result)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('index.html', error=error_message)

    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
