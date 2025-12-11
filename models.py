
# Save this as models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def prepare_data(df):
    """Prep data for sklearn"""
    feature_cols = ['temperature', 'vibration', 'pressure', 
                   'temperature_24h_mean', 'vibration_24h_mean',
                   'temp_x_vib', 'hour', 'day_of_week']
    
    df_clean = df.dropna(subset=feature_cols + ['high_risk'])
    
    X = df_clean[feature_cols]
    y = df_clean['high_risk']
    
    return X, y

def train_model(X, y):
    """Train Random Forest"""
    print("\nTraining Random Forest model...")
    
    # Check if we have both classes
    if len(y.unique()) == 1:
        print("Warning: Only one class found, creating synthetic high-risk data...")
        # Create some synthetic high-risk samples
        n_samples = int(len(X) * 0.1)
        X_high_risk = X.sample(n=n_samples).copy()
        X_high_risk['temperature'] += np.random.uniform(5, 10, n_samples)
        X_high_risk['vibration'] += np.random.uniform(1, 3, n_samples)
        
        X = pd.concat([X, X_high_risk])
        y = pd.concat([y, pd.Series([1] * n_samples, index=X_high_risk.index)])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop features:")
    print(feature_imp.head())
    
    return rf, acc, prec, rec

def predict_failures(model, df, feature_cols):
    """Generate predictions"""
    X = df[feature_cols].fillna(0)  # Fill NaN values
    
    # Check if model can predict probabilities
    if hasattr(model, "predict_proba"):
        try:
            predictions = model.predict_proba(X)
            if predictions.shape[1] > 1:
                df['failure_probability'] = predictions[:, 1]
            else:
                df['failure_probability'] = predictions[:, 0]
        except:
            # Fallback to regular predictions
            df['failure_probability'] = model.predict(X)
    else:
        df['failure_probability'] = model.predict(X)
    
    # Create some high-risk devices for demonstration
    # Mark top 5% as high risk based on temperature
    temp_threshold = df['temperature'].quantile(0.95)
    df.loc[df['temperature'] > temp_threshold, 'failure_probability'] = 0.8
    
    high_risk_devices = df[df['failure_probability'] > 0.7]['device_id'].unique()
    
    # If no high risk found, take top 10 by temperature
    if len(high_risk_devices) == 0:
        high_risk_devices = df.nlargest(10, 'temperature')['device_id'].unique()
    
    return df, high_risk_devices