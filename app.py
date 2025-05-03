from flask import Flask, render_template, request, redirect, url_for
import os
import pickle
from improved_model import prepare_input_data

app = Flask(__name__)

def load_improved_model(model_path='saved_model/liver_disease_model.pkl',
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
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        # Try to use the old model if improved model is not available
        if os.path.exists('saved_model/liver_disease_model.pkl') and os.path.exists('saved_model/scaler.pkl'):
            with open('saved_model/liver_disease_model.pkl', 'rb') as f:
                model = pickle.load(f)

            with open('saved_model/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

            # Create a default feature names list for the old model
            feature_names = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                           'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                           'Aspartate_Aminotransferase', 'Total_Protiens',
                           'Albumin', 'Albumin_and_Globulin_Ratio']

            return model, scaler, feature_names
        else:
            raise FileNotFoundError(f"Model or scaler not found. Please train the model first by running improved_model.py.")

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
    except:
        raise FileNotFoundError(f"Error loading model files. Please train the model first by running improved_model.py.")

    return model, scaler, feature_names

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
            # Get form data
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

            # Load model, scaler, and feature names
            model, scaler, feature_names = load_improved_model()

            # Make prediction
            prediction, probability, severity = predict_liver_disease(patient_data, model, scaler, feature_names)

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
