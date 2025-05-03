import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(data):
    """
    Explore the dataset and print summary statistics
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to explore
    """
    print("Dataset Shape:", data.shape)
    print("\nFirst 5 rows:")
    print(data.head())
    
    print("\nData Types:")
    print(data.dtypes)
    
    print("\nSummary Statistics:")
    print(data.describe())
    
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    print("\nClass Distribution:")
    print(data['Dataset'].value_counts())
    print(f"Class 1 (Liver Disease): {data['Dataset'].value_counts()[1]} ({data['Dataset'].value_counts()[1]/len(data)*100:.2f}%)")
    print(f"Class 2 (No Liver Disease): {data['Dataset'].value_counts()[2] if 2 in data['Dataset'].value_counts() else 0} ({data['Dataset'].value_counts()[2]/len(data)*100 if 2 in data['Dataset'].value_counts() else 0:.2f}%)")

def plot_distributions(data):
    """
    Plot distributions of features
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to plot
    """
    # Set up the matplotlib figure
    plt.figure(figsize=(15, 10))
    
    # Get numerical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols.remove('Dataset')  # Remove target variable
    
    # Plot histograms
    for i, col in enumerate(numerical_cols):
        plt.subplot(3, 3, i+1)
        sns.histplot(data=data, x=col, hue='Dataset', kde=True, bins=20)
        plt.title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data):
    """
    Plot correlation matrix
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to plot
    """
    # Calculate correlation matrix
    corr = data.corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Draw the heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def plot_boxplots(data):
    """
    Plot boxplots for each feature by target class
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to plot
    """
    # Get numerical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols.remove('Dataset')  # Remove target variable
    
    # Set up the matplotlib figure
    plt.figure(figsize=(15, 15))
    
    # Plot boxplots
    for i, col in enumerate(numerical_cols):
        plt.subplot(3, 3, i+1)
        sns.boxplot(x='Dataset', y=col, data=data)
        plt.title(f'Boxplot of {col} by Target Class')
    
    plt.tight_layout()
    plt.show()

def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print classification report with custom formatting
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    target_names : list, optional
        List of target class names
    """
    from sklearn.metrics import classification_report
    
    if target_names is None:
        target_names = ['No Disease', 'Disease']
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\nClassification Report:")
    print(report)
