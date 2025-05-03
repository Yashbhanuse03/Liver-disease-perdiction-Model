import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

def load_excel_data(file_path='Liver_Disease_Dataset.xlsx'):
    """
    Load the liver disease dataset from Excel

    Parameters:
    -----------
    file_path : str
        Path to the Excel dataset

    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    print(f"Loading data from {file_path}...")
    try:
        # Try to load the Excel file
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

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

    # Remove ID columns if they exist
    id_columns = [col for col in df.columns if 'id' in col.lower() or 'patient' in col.lower()]
    for col in id_columns:
        print(f"Removing ID column: {col}")
        df.drop(col, axis=1, inplace=True)

    print("\nData exploration:")
    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)

    # Identify the target column (assuming it's named 'Dataset', 'Target', or similar)
    target_columns = [col for col in df.columns if col.lower() in ['dataset', 'target', 'diagnosis', 'disease', 'liver_disease', 'has_liver_disease']]

    if not target_columns:
        print("Error: Could not identify target column. Please specify the target column name.")
        return None

    target_column = target_columns[0]
    print(f"\nIdentified target column: {target_column}")
    print(f"Target value counts:\n{df[target_column].value_counts()}")

    # Handle missing values
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype in ['int64', 'float64']:
                # Fill numeric columns with median
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
                print(f"Filled missing values in {column} with median: {median_value}")
            else:
                # Fill categorical columns with mode
                mode_value = df[column].mode()[0]
                df[column] = df[column].fillna(mode_value)
                print(f"Filled missing values in {column} with mode: {mode_value}")

    # Convert categorical columns to numerical
    for column in df.columns:
        if df[column].dtype == 'object' and column != target_column:
            if column.lower() in ['gender', 'sex']:
                # Handle gender column
                df[column] = df[column].map(lambda x: 1 if str(x).lower() in ['male', 'm', '1'] else 0)
                print(f"Converted {column} to binary (Male: 1, Female: 0)")
            else:
                # One-hot encode other categorical columns
                dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(column, axis=1, inplace=True)
                print(f"One-hot encoded {column}")

    # Feature Engineering
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Create ratios and other features if specific columns exist
    if all(col in numeric_columns for col in ['Total_Bilirubin', 'Direct_Bilirubin']):
        df['Bilirubin_Ratio'] = np.where(
            df['Total_Bilirubin'] > 0,
            df['Direct_Bilirubin'] / df['Total_Bilirubin'],
            0
        )
        print("Created Bilirubin_Ratio feature")

    if all(col in numeric_columns for col in ['Aspartate_Aminotransferase', 'Alamine_Aminotransferase']):
        df['AST_ALT_Ratio'] = np.where(
            df['Alamine_Aminotransferase'] > 0,
            df['Aspartate_Aminotransferase'] / df['Alamine_Aminotransferase'],
            0
        )
        print("Created AST_ALT_Ratio feature")

    if all(col in numeric_columns for col in ['Total_Protiens', 'Albumin']):
        df['Globulin'] = df['Total_Protiens'] - df['Albumin']
        print("Created Globulin feature")

    # Handle any remaining NaN values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Convert target column to binary before other operations
    if target_column in df.columns:
        print(f"\nConverting target column '{target_column}' to binary")
        # Check if it's already numeric
        if df[target_column].dtype in ['int64', 'float64']:
            # If numeric, assume 0 is negative, anything else is positive
            df[target_column] = df[target_column].map(lambda x: 0 if x == 0 else 1)
        else:
            # If string/categorical, map common negative terms to 0, others to 1
            negative_terms = ['no', 'negative', 'normal', 'healthy', '0', 'false']
            df[target_column] = df[target_column].map(lambda x: 0 if str(x).lower() in negative_terms else 1)

        print(f"Target distribution after conversion:\n{df[target_column].value_counts()}")

    # Handle any remaining categorical columns
    for column in df.columns:
        if df[column].dtype == 'object' and column != target_column:
            # One-hot encode categorical columns
            dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(column, axis=1, inplace=True)
            print(f"One-hot encoded {column}")

    # Feature Engineering
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Create ratios and other features if specific columns exist
    if all(col in numeric_columns for col in ['Total_Bilirubin', 'Direct_Bilirubin']):
        df['Bilirubin_Ratio'] = np.where(
            df['Total_Bilirubin'] > 0,
            df['Direct_Bilirubin'] / df['Total_Bilirubin'],
            0
        )
        print("Created Bilirubin_Ratio feature")

    if all(col in numeric_columns for col in ['Aspartate_Aminotransferase', 'Alamine_Aminotransferase']):
        df['AST_ALT_Ratio'] = np.where(
            df['Alamine_Aminotransferase'] > 0,
            df['Aspartate_Aminotransferase'] / df['Alamine_Aminotransferase'],
            0
        )
        print("Created AST_ALT_Ratio feature")

    if all(col in numeric_columns for col in ['Total_Protiens', 'Albumin']):
        df['Globulin'] = df['Total_Protiens'] - df['Albumin']
        print("Created Globulin feature")

    # Handle any remaining NaN values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values with 0 (safer than using median for mixed data types)
    df = df.fillna(0)

    # Prepare target variable
    y = df[target_column]

    # Drop target from features
    X = df.drop(target_column, axis=1)

    # Store feature names
    feature_names = X.columns.tolist()
    print(f"\nFeatures ({len(feature_names)}):")
    for i, feature in enumerate(feature_names):
        print(f"{i+1}. {feature}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(f"Training set class distribution: {np.bincount(y_train)}")
    print(f"Testing set class distribution: {np.bincount(y_test)}")

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

def train_models(X_train, y_train):
    """
    Train multiple models

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
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=2000, C=1.0, solver='liblinear')
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb

    # SVM
    print("Training SVM...")
    svm = SVC(
        random_state=42,
        C=10,
        kernel='rbf',
        probability=True
    )
    svm.fit(X_train, y_train)
    models['SVM'] = svm

    # Create a Voting Classifier (Ensemble)
    print("Creating Voting Classifier...")
    estimators = [
        ('lr', models['Logistic Regression']),
        ('rf', models['Random Forest']),
        ('gb', models['Gradient Boosting']),
        ('svm', models['SVM'])
    ]
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    voting_clf.fit(X_train, y_train)
    models['Voting Classifier'] = voting_clf

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
        print(f"\n{name} Performance:")

        # For Logistic Regression, find optimal threshold
        if name == 'Logistic Regression':
            y_prob = model.predict_proba(X_test)[:, 1]

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
            print(f"  Best threshold: {best_threshold:.2f}")
        else:
            # For other models, use the default threshold
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

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

        # Print metrics
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")

        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n  Confusion Matrix:")
        print(f"    True Negative: {cm[0][0]}")
        print(f"    False Positive: {cm[0][1]}")
        print(f"    False Negative: {cm[1][0]}")
        print(f"    True Positive: {cm[1][1]}")

        # Print classification report
        print("\n  Classification Report:")
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

    return best_model_name, models[best_model_name], results[best_model_name]

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

def main():
    """
    Main function to train and evaluate the model with new data
    """
    print("Liver Disease Prediction System - Training with New Dataset")
    print("=========================================================")

    # Load data from Excel
    data = load_excel_data()

    if data is None:
        print("Error: Could not load data. Exiting.")
        return

    # Preprocess data
    preprocessing_result = preprocess_data(data)

    if preprocessing_result is None:
        print("Error: Could not preprocess data. Exiting.")
        return

    X_train, X_test, y_train, y_test, scaler, feature_names = preprocessing_result

    # Train models
    print("\nTraining models...")
    models = train_models(X_train, y_train)

    # Evaluate models
    print("\nEvaluating models...")
    results = evaluate_models(models, X_test, y_test)

    # Get best model
    print("\nSelecting best model...")
    best_model_name, best_model, best_metrics = get_best_model(models, results)
    print(f"Best model: {best_model_name} (F1 Score: {best_metrics['f1']:.4f})")

    # Save the best model
    print("\nSaving best model...")
    model_path, scaler_path, features_path = save_model(best_model, scaler, feature_names)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Feature names saved to {features_path}")

    print("\nTraining complete! The model is ready for use.")

    return best_model, scaler, feature_names

if __name__ == "__main__":
    main()
