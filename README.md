# Liver Disease Prediction System

This system predicts the likelihood of liver disease based on patient medical data using machine learning algorithms.

## Dataset

The system uses the Indian Liver Patient Dataset (ILPD) which contains the following features:

- Age: Age of the patient
- Gender: Gender of the patient (Male/Female)
- Total_Bilirubin: Total bilirubin level
- Direct_Bilirubin: Direct bilirubin level
- Alkaline_Phosphotase: Alkaline phosphatase level
- Alamine_Aminotransferase: Alamine aminotransferase level
- Aspartate_Aminotransferase: Aspartate aminotransferase level
- Total_Protiens: Total proteins level
- Albumin: Albumin level
- Albumin_and_Globulin_Ratio: Albumin and globulin ratio
- Dataset: Target variable (1: Liver disease, 2: No liver disease)

## System Components

1. **preprocessing.py**: Contains functions for data loading and preprocessing
2. **model.py**: Contains functions for model training, evaluation, and visualization
3. **utils.py**: Contains utility functions for data exploration and visualization
4. **main.py**: Main script to train and evaluate models
5. **predict.py**: Script for making predictions on new patient data

## How to Use

### Training the Model

To train the model and evaluate its performance:

```bash
python main.py
```

This will:
- Load and preprocess the data
- Train multiple machine learning models (Logistic Regression, Random Forest, SVM, Gradient Boosting)
- Evaluate the models and select the best one
- Save the best model for future use

### Making Predictions

To make predictions for new patients:

```bash
python predict.py
```

For interactive mode where you can input patient data:

```bash
python predict.py --interactive
```

## Model Performance

The system evaluates models using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

The best model is selected based on the F1 score, which balances precision and recall.

## Requirements

The system requires the following Python packages:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib

You can install these packages using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```
