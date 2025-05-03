import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score   
from sklearn.metrics import confusion_matrix, classification_report
import os
import pickle

def train_models(X_train, y_train):    
    """
    Train multiple models and return them

    Parameters:
    -----------
    X_train : np.array
        Training features
    y_train : np.array
        Training target

    Returns:
    --------
    dict
        Dictionary containing trained models
    """
    models = {}

    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate models on test data

    Parameters:
    -----------
    models : dict
        Dictionary containing trained models
    X_test : np.array
        Test features
    y_test : np.array
        Test target

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics for each model
    """
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

    return results

def get_best_model(models, results):
    """
    Get the best model based on F1 score

    Parameters:
    -----------
    models : dict
        Dictionary containing trained models
    results : dict
        Dictionary containing evaluation metrics

    Returns:
    --------
    tuple
        Best model name, model object, and its metrics
    """
    best_f1 = 0
    best_model_name = None

    for name, metrics in results.items():
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_name = name

    return best_model_name, models[best_model_name], results[best_model_name]

def save_model(model, scaler, model_dir='saved_model'):
    """
    Save the model and scaler to disk

    Parameters:
    -----------
    model : object
        Trained model to save
    scaler : object
        Fitted scaler to save
    model_dir : str
        Directory to save the model

    Returns:
    --------
    tuple
        Paths to the saved model and scaler
    """
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save model and scaler
    model_path = os.path.join(model_dir, 'liver_disease_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    return model_path, scaler_path

def load_model(model_path, scaler_path):
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
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler

def print_confusion_matrix(y_test, y_pred, title='Confusion Matrix'):
    """
    Print confusion matrix

    Parameters:
    -----------
    y_test : np.array
        True labels
    y_pred : np.array
        Predicted labels
    title : str
        Plot title
    """
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{title}:")
    print(f"True Negative: {cm[0][0]}")
    print(f"False Positive: {cm[0][1]}")
    print(f"False Negative: {cm[1][0]}")
    print(f"True Positive: {cm[1][1]}")

def print_feature_importance(model, feature_names):
    """
    Print feature importance for tree-based models

    Parameters:
    -----------
    model : object
        Trained model
    feature_names : list
        List of feature names
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nFeature Importance:")
        for i in range(len(importances)):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    else:
        print("\nThis model doesn't support feature importance visualization.")
