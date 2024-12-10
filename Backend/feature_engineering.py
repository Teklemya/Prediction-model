import pandas as pd
import numpy as np

def create_features(df):
    """
    Engineer features for NFL game prediction model
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with NFL game data
    
    Returns:
    pandas.DataFrame: DataFrame with engineered features
    """
    # Create a copy to avoid modifying the original DataFrame
    df_processed = df.copy()
    
    # Create a binary target variable (1 if home team wins, 0 otherwise)
    df_processed['home_team_win'] = (df_processed['score_home'] > df_processed['score_away']).astype(int)
    
    # Extract additional features
    df_processed['point_difference'] = df_processed['score_home'] - df_processed['score_away']
    
    # Convert date to datetime
    df_processed['schedule_date'] = pd.to_datetime(df_processed['schedule_date'])
    
    # Extract month and season
    df_processed['month'] = df_processed['schedule_date'].dt.month
    df_processed['year'] = df_processed['schedule_date'].dt.year
    
    # One-hot encode categorical variables
    df_processed = pd.get_dummies(df_processed, columns=['team_home', 'team_away'])
    
    # Handle weather features
    weather_columns = ['weather_temperature', 'weather_wind_mph', 'weather_humidity']
    for col in weather_columns:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    
    return df_processed