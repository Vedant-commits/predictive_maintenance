# feature_engineering.py
# Feature engineering based on domain knowledge

import pandas as pd
import numpy as np

def add_rolling_features(df):
    """
    Adding rolling statistics - found these work well for equipment data
    """
    print("Calculating rolling features...")
    
    # Sort by device and time first
    df = df.sort_values(['device_id', 'timestamp'])
    
    # 24-hour rolling stats (1 day window)
    for col in ['temperature', 'vibration']:
        # Rolling mean
        df[f'{col}_24h_mean'] = df.groupby('device_id')[col].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )
        
        # Rate of change 
        df[f'{col}_change'] = df.groupby('device_id')[col].diff()
    
    # Mark anomalies (simple threshold for now)
    df['temp_anomaly'] = 0
    df.loc[df['temperature'] > df['temperature_24h_mean'] + 5, 'temp_anomaly'] = 1
    
    return df

def calculate_health_score(df):
    """
    Simple health score based on sensor readings
    TODO: Refine this with actual failure data
    """
    # Normalize each metric
    df['temp_score'] = 100 - np.abs(df['temperature'] - 22) * 2  # 22 is ideal
    df['vib_score'] = 100 - df['vibration'] * 10  # Lower vibration is better
    
    # Clip to 0-100 range
    df['temp_score'] = df['temp_score'].clip(0, 100)
    df['vib_score'] = df['vib_score'].clip(0, 100)
    
    # Combined health score
    df['health_score'] = (df['temp_score'] + df['vib_score']) / 2
    
    # Add failure risk flag
    df['high_risk'] = (df['health_score'] < 70).astype(int)
    
    return df

def create_ml_features(df):
    """
    Features for ML model
    """
    df = add_rolling_features(df)
    df = calculate_health_score(df)
    
    # Interaction features
    df['temp_x_vib'] = df['temperature'] * df['vibration']
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    print(f"Total features: {len(df.columns)}")
    return df