import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

def preprocess_data(data, test_size=0.2, random_state=42):
    """
    Preprocess the liver disease dataset

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to preprocess
    test_size : float
        Size of the test set
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X_train, X_test, y_train, y_test, scaler
        Preprocessed training and testing data, and the scaler object
    """
    # Make a copy of the data
    df = data.copy()

    # Handle missing values in 'Albumin_and_Globulin_Ratio'
    df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median(), inplace=True)

    # Convert Gender to numerical (Male: 1, Female: 0)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Separate features and target
    X = df.drop('Dataset', axis=1)
    y = df['Dataset']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def prepare_input_data(patient_data, scaler):
    """
    Prepare input data for prediction

    Parameters:
    -----------
    patient_data : dict
        Dictionary containing patient information
    scaler : StandardScaler
        Scaler used to scale the training data

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

    # Scale the data
    scaled_data = scaler.transform(df)

    return scaled_data
