# Import libraries used in these functions

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import pandas as pd


# Function to evaluate classification models - 1

def evaluate_classification_models(models, cv, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates multiple machine learning models,and returns a DataFrame summarizing the performance metrics of each model.

    For each model, it calculates the following metrics:
    - Accuracy (calculated using cross-validation)
    - Precision
    - Recall
    - F1 Score

    Parameters:
    models: A dictionary containing the models to be trained
    cv (int): The number of folds to use for cross-validation.
    X_train: Training features
    y_train: Training labels
    X_test: Testing features
    y_test: Testing labels

    Returns:
    pd.DataFrame: A DataFrame containing the performance metrics for each model.
                  The DataFrame has columns for the model name, accuracy, precision,
                  recall, F1 score, and Cross Validation Score.
    """
   
    # Function implementation below


    metrics_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "CV Accuracy"])

    for model_name, model_class in models.items():
        # Instantiate the model
        model = model_class()

        # Train the model on training dataset
        model.fit(X_train, y_train)
    
        # Predict values of y
        y_pred = model.predict(X_test)

        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)

        accuracy = f"{report["accuracy"]*100:.2f}%"
        precision = report["macro avg"]["precision"]
        recall = report["macro avg"]["recall"]
        f1_score = report["macro avg"]["f1-score"] 

        # Cross Validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
        cv_accuracy = f"{cv_scores.mean()*100:.2f}%"

        model_metrics = pd.DataFrame([[model_name, accuracy, precision,recall, f1_score, cv_accuracy]], columns=metrics_df.columns)

        if metrics_df.empty:
            metrics_df = model_metrics
        else:
            # Append result to metrics dataframe
            metrics_df = pd.concat([metrics_df, model_metrics], ignore_index=True)

    return metrics_df




# Function to evaluate classification models - 2
# makes uses of pipeline

def evaluate_classification_models_2(models, cv, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates multiple machine learning models, and returns a DataFrame summarizing the performance metrics of each model.

    Parameters:
    models: A dictionary containing the models to be trained
    cv (int): The number of folds to use for cross-validation.
    X_train: Training features
    y_train: Training labels
    X_test: Testing features
    y_test: Testing labels

    Returns:
    pd.DataFrame: A DataFrame containing the performance metrics for each model.
                  The DataFrame has columns for the model name, accuracy, precision,
                  recall, F1 score, and cross-validation score.
    """
    metrics_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "CV Accuracy"])

    for model_name, model_class in models.items():
        # Create a pipeline (optional, but good for consistency)
        model_pipeline = make_pipeline(model_class())
        
        try:
            # Train the model
            model_pipeline.fit(X_train, y_train)
            
            # Predict on the test set
            y_pred = model_pipeline.predict(X_test)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)

            accuracy = f"{report['accuracy']*100:.2f}%"
            precision = report["macro avg"]["precision"]
            recall = report["macro avg"]["recall"]
            f1_score = report["macro avg"]["f1-score"]

            # Cross-validation score
            cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=cv)
            cv_accuracy = f"{cv_scores.mean()*100:.2f}%"

            model_metrics = pd.DataFrame([[model_name, accuracy, precision, recall, f1_score, cv_accuracy]], columns=metrics_df.columns)

            # Append result to metrics dataframe
            if metrics_df.empty:
                metrics_df = model_metrics
            else:
                metrics_df = pd.concat([metrics_df, model_metrics], ignore_index=True)
        
        except Exception as e:
            print(f"An error occurred while processing {model_name}: {e}")

    return metrics_df

    