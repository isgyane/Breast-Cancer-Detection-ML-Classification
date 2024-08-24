# Import libraries used in these functions

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


# Function to evaluate classification models - 1st Approach
# Without a pipeline

def evaluate_classification_models_1(models, fold, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates multiple machine learning models, and returns a DataFrame summarizing the performance metrics of each model.

    Parameters:
    models: A dictionary containing the models to be trained
    fold (int): The number of folds to use for cross-validation.
    X_train: Training features
    y_train: Training labels
    X_test: Testing features
    y_test: Testing labels

    Returns:
    pd.DataFrame: A DataFrame containing the performance metrics for each model.
                  The DataFrame has columns for the model name, accuracy, precision,
                  recall, F1 score, cross-validation accuracy, and stratified cross-validation accuracy.
    """

    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize DataFrame to store results
    metrics_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "CV Accuracy", "SCV Accuracy"])
    
    # Loop through models
    for model_name, model_class in models.items():
        try:
            # Instantiate the model
            model = model_class()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)

            ## Extract metrics of the classification report
            accuracy = f"{report['accuracy']*100:.2f}%"
            precision = report["macro avg"]["precision"]
            recall = report["macro avg"]["recall"]
            f1_score = report["macro avg"]["f1-score"]

            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=fold, scoring="accuracy")
            cv_accuracy = f"{cv_scores.mean()*100:.2f}%"

            # Stratified Crossvalidation
            skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
            scv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
            scv_accuracy = f"{scv_scores.mean()*100:.2f}%"

            # Compile results
            model_metrics = pd.DataFrame([[model_name, accuracy, precision, recall, f1_score, cv_accuracy, scv_accuracy]], columns=metrics_df.columns)

            # Append result to metrics dataframe
            if metrics_df.empty:
                metrics_df = model_metrics
            else:
                metrics_df = pd.concat([metrics_df, model_metrics], ignore_index=True)
        
        except Exception as e:
            print(f"An error occurred while processing {model_name}: {e}")

    return metrics_df




# Function to evaluate classification models - Second Approach
# makes uses of pipeline

def evaluate_classification_models_2(models, fold, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates multiple machine learning models, and returns a DataFrame summarizing the performance metrics of each model.

    Parameters:
    models: A dictionary containing the models to be trained
    fold (int): The number of folds to use for cross-validation.
    X_train: Training features
    y_train: Training labels
    X_test: Testing features
    y_test: Testing labels

    Returns:
    pd.DataFrame: A DataFrame containing the performance metrics for each model.
                  The DataFrame has columns for the model name, accuracy, precision,
                  recall, F1 score, cross-validation accuracy, and stratified cross-validation accuracy.
    """

    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize DataFrame to store results
    metrics_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "CV Accuracy", "SCV Accuracy"])
    
    # Loop through models
    for model_name, model_class in models.items():
        # Create a pipeline
        model_pipeline = make_pipeline(model_class())
        
        try:
            # Train model
            model_pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model_pipeline.predict(X_test)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)

            ## Extract metrics of the classification report
            accuracy = f"{report['accuracy']*100:.2f}%"
            precision = report["macro avg"]["precision"]
            recall = report["macro avg"]["recall"]
            f1_score = report["macro avg"]["f1-score"]

            # Cross-validation score
            cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=fold, scoring="accuracy")
            cv_accuracy = f"{cv_scores.mean()*100:.2f}%"

            # Stratified Crossvalidation
            skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
            scv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=skf, scoring="accuracy")
            scv_accuracy = f"{scv_scores.mean()*100:.2f}%"

            # Compile results
            model_metrics = pd.DataFrame([[model_name, accuracy, precision, recall, f1_score, cv_accuracy, scv_accuracy]], columns=metrics_df.columns)

            # Append result to metrics dataframe
            if metrics_df.empty:
                metrics_df = model_metrics
            else:
                metrics_df = pd.concat([metrics_df, model_metrics], ignore_index=True)
        
        except Exception as e:
            print(f"An error occurred while processing {model_name}: {e}")

    return metrics_df
    

    