import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

def prepare_data(df, test_size=0.2, random_state=42):
    """
    Prepare data for model training
    
    Parameters:
    df (pandas.DataFrame): Processed DataFrame
    test_size (float): Proportion of data to use for testing
    random_state (int): Seed for reproducibility
    
    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    # Determine features and target
    features = [col for col in df.columns if col not in ['home_team_win', 'schedule_date']]
    X = df[features]
    y = df['home_team_win']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Classifier
    
    Parameters:
    X_train (array): Scaled training features
    y_train (array): Training target
    n_estimators (int): Number of trees in the forest
    random_state (int): Seed for reproducibility
    
    Returns:
    RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state,
        # Additional parameters for better performance
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Parameters:
    model (RandomForestClassifier): Trained model
    X_test (array): Scaled test features
    y_test (array): Test target
    
    Returns:
    dict: Performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Print results
    print("Model Performance Metrics:")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    # Feature importance
    feature_importance = model.feature_importances_
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'feature_importance': feature_importance
    }

def get_feature_importance(model, feature_names):
    """
    Get and display feature importances
    
    Parameters:
    model (RandomForestClassifier): Trained model
    feature_names (list): Names of features
    
    Returns:
    pandas.DataFrame: Sorted feature importances
    """
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_imp.head(10))
    
    return feature_imp