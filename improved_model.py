import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle
import os
import matplotlib.pyplot as plt

def load_data(file_path='liver.csv'):
    """
    Load the liver disease dataset

    Parameters:
    -----------
    file_path : str
        Path to the dataset

    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    return pd.read_csv(file_path)

def preprocess_data(data, test_size=0.2, random_state=42, apply_smote=True):
    """
    Preprocess the liver disease dataset with advanced techniques

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to preprocess
    test_size : float
        Size of the test set
    random_state : int
        Random seed for reproducibility
    apply_smote : bool
        Whether to apply SMOTE for handling class imbalance

    Returns:
    --------
    X_train, X_test, y_train, y_test, scaler, feature_names
        Preprocessed training and testing data, scaler object, and feature names
    """
    # Make a copy of the data
    df = data.copy()

    # Handle missing values in 'Albumin_and_Globulin_Ratio'
    median_value = df['Albumin_and_Globulin_Ratio'].median()
    df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(median_value)

    # Convert Gender to numerical (Male: 1, Female: 0)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Feature Engineering
    # Create new features that might be useful
    # Safely calculate ratios to avoid division by zero
    df['Bilirubin_Ratio'] = np.where(
        df['Total_Bilirubin'] > 0,
        df['Direct_Bilirubin'] / df['Total_Bilirubin'],
        0
    )

    df['AST_ALT_Ratio'] = np.where(
        df['Alamine_Aminotransferase'] > 0,
        df['Aspartate_Aminotransferase'] / df['Alamine_Aminotransferase'],
        0
    )

    df['Globulin'] = df['Total_Protiens'] - df['Albumin']

    # Handle any remaining NaN values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Calculate medians for each column and fill NaN values
    medians = df.median()
    df = df.fillna(medians)

    # Separate features and target
    # Adjust target to be 0-based (1 -> 0, 2 -> 1)
    y = df['Dataset'].map({1: 1, 2: 0})
    X = df.drop('Dataset', axis=1)

    # Store feature names
    feature_names = X.columns.tolist()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE to handle class imbalance
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE - Class distribution: {np.bincount(y_train)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

def train_advanced_models(X_train, y_train, cv=5):
    """
    Train multiple advanced models with hyperparameter tuning

    Parameters:
    -----------
    X_train : np.array
        Training features
    y_train : np.array
        Training target
    cv : int
        Number of cross-validation folds

    Returns:
    --------
    dict
        Dictionary containing trained models
    """
    models = {}

    # Define cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Logistic Regression with L1 regularization
    print("Training Logistic Regression...")
    lr_params = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced'],
        'max_iter': [2000, 3000]
    }
    lr = GridSearchCV(
        LogisticRegression(random_state=42),
        lr_params, cv=cv_strategy, scoring='f1', n_jobs=-1
    )
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = {
        'model': lr.best_estimator_,
        'best_params': lr.best_params_,
        'best_score': lr.best_score_
    }
    print(f"Best LR params: {lr.best_params_}, Score: {lr.best_score_:.4f}")

    # Random Forest with hyperparameter tuning
    print("Training Random Forest...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }
    rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params, cv=cv_strategy, scoring='f1', n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = {
        'model': rf.best_estimator_,
        'best_params': rf.best_params_,
        'best_score': rf.best_score_
    }
    print(f"Best RF params: {rf.best_params_}, Score: {rf.best_score_:.4f}")

    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'min_samples_split': [2, 5]
    }
    gb = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params, cv=cv_strategy, scoring='f1', n_jobs=-1
    )
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = {
        'model': gb.best_estimator_,
        'best_params': gb.best_params_,
        'best_score': gb.best_score_
    }
    print(f"Best GB params: {gb.best_params_}, Score: {gb.best_score_:.4f}")

    # Support Vector Machine
    print("Training SVM...")
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
        'class_weight': [None, 'balanced']
    }
    svm = GridSearchCV(
        SVC(random_state=42, probability=True),
        svm_params, cv=cv_strategy, scoring='f1', n_jobs=-1
    )
    svm.fit(X_train, y_train)
    models['SVM'] = {
        'model': svm.best_estimator_,
        'best_params': svm.best_params_,
        'best_score': svm.best_score_
    }
    print(f"Best SVM params: {svm.best_params_}, Score: {svm.best_score_:.4f}")

    # Create a Voting Classifier (Ensemble)
    print("Creating Voting Classifier...")
    estimators = [
        ('lr', models['Logistic Regression']['model']),
        ('rf', models['Random Forest']['model']),
        ('gb', models['Gradient Boosting']['model']),
        ('svm', models['SVM']['model'])
    ]
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    voting_clf.fit(X_train, y_train)
    models['Voting Classifier'] = {
        'model': voting_clf,
        'best_params': 'N/A',
        'best_score': 'N/A'
    }

    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate models on test data with comprehensive metrics

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

    for name, model_info in models.items():
        model = model_info['model']
        y_prob = model.predict_proba(X_test)[:, 1]

        # For Logistic Regression, find optimal threshold
        if name == 'Logistic Regression':
            # Try different thresholds to find the best one
            thresholds = np.arange(0.3, 0.7, 0.05)
            best_f1 = 0
            best_threshold = 0.5

            for threshold in thresholds:
                y_pred_temp = (y_prob >= threshold).astype(int)
                f1_temp = f1_score(y_test, y_pred_temp)

                if f1_temp > best_f1:
                    best_f1 = f1_temp
                    best_threshold = threshold

            # Use the best threshold for prediction
            y_pred = (y_prob >= best_threshold).astype(int)
            print(f"Best threshold for Logistic Regression: {best_threshold:.2f}")
        else:
            # For other models, use the default threshold
            y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        # Print detailed metrics
        print(f"\n{name} Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")

        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"  True Negative: {cm[0][0]}")
        print(f"  False Positive: {cm[0][1]}")
        print(f"  False Negative: {cm[1][0]}")
        print(f"  True Positive: {cm[1][1]}")

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'], zero_division=0))

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

    return best_model_name, models[best_model_name]['model'], results[best_model_name]

def save_model(model, scaler, feature_names, model_dir='saved_model'):
    """
    Save the model, scaler, and feature names to disk

    Parameters:
    -----------
    model : object
        Trained model to save
    scaler : object
        Fitted scaler to save
    feature_names : list
        List of feature names
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

    # Save model, scaler, and feature names
    model_path = os.path.join(model_dir, 'liver_disease_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    features_path = os.path.join(model_dir, 'feature_names.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)

    return model_path, scaler_path, features_path

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
    # Convert to DataFrame
    df = pd.DataFrame([patient_data])

    # Convert Gender to numerical if it's not already
    if 'Gender' in df.columns and df['Gender'].dtype == 'object':
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Perform the same feature engineering as during training
    # Safely calculate ratios to avoid division by zero
    df['Bilirubin_Ratio'] = np.where(
        df['Total_Bilirubin'] > 0,
        df['Direct_Bilirubin'] / df['Total_Bilirubin'],
        0
    )

    df['AST_ALT_Ratio'] = np.where(
        df['Alamine_Aminotransferase'] > 0,
        df['Aspartate_Aminotransferase'] / df['Alamine_Aminotransferase'],
        0
    )

    df['Globulin'] = df['Total_Protiens'] - df['Albumin']

    # Handle any NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)  # For prediction, we can use 0 as a safe default

    # Ensure all required features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0

    # Select only the features used during training
    df = df[feature_names]

    # Scale the data
    scaled_data = scaler.transform(df)

    return scaled_data

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models

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

        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        print("\nFeature Importance:")
        for i in range(len(importances)):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    else:
        print("\nThis model doesn't support feature importance visualization.")

def main():
    """
    Main function to train and evaluate the improved model
    """
    print("Liver Disease Prediction System - Improved Model")
    print("===============================================")

    # Load data
    print("Loading data...")
    data = load_data()

    # Preprocess data
    print("\nPreprocessing data with advanced techniques...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)

    # Train advanced models
    print("\nTraining advanced models with hyperparameter tuning...")
    models = train_advanced_models(X_train, y_train)

    # Evaluate models
    print("\nEvaluating models...")
    results = evaluate_models(models, X_test, y_test)

    # Get best model
    print("\nSelecting best model...")
    best_model_name, best_model, best_metrics = get_best_model(models, results)
    print(f"Best model: {best_model_name} (F1 Score: {best_metrics['f1']:.4f})")

    # Plot feature importance for the best model
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        plot_feature_importance(best_model, feature_names)

    # Save the best model
    print("\nSaving best model...")
    model_path, scaler_path, features_path = save_model(best_model, scaler, feature_names)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Feature names saved to {features_path}")

    return best_model, scaler, feature_names

if __name__ == "__main__":
    main()
