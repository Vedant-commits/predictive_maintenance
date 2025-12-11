# main.py
# Main execution script
# Author: [Your name]
# Date: [Today]

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from data_generator import create_sensor_data
from feature_engineering import create_ml_features
from models import prepare_data, train_model, predict_failures
from visualizations import plot_sensor_trends, plot_risk_distribution, create_executive_summary

def main():
    """
    Run the complete pipeline
    """
    print("="*50)
    print("MEDICAL EQUIPMENT PREDICTIVE MAINTENANCE")
    print("="*50)
    
    # Step 1: Generate data
    print("\n1. Generating sensor data...")
    df = create_sensor_data()
    
    # Step 2: Feature engineering  
    print("\n2. Engineering features...")
    df = create_ml_features(df)
    
    # Step 3: Train model
    print("\n3. Training ML model...")
    X, y = prepare_data(df)
    model, acc, prec, rec = train_model(X, y)
    
    # Step 4: Generate predictions
    print("\n4. Generating predictions...")
    feature_cols = X.columns.tolist()
    df, high_risk = predict_failures(model, df, feature_cols)
    
    print(f"\nFound {len(high_risk)} high-risk devices")
    print("High risk devices:", high_risk[:5])  # Show first 5
    
    # Step 5: Create visualizations
    print("\n5. Creating visualizations...")
    
    # Plot for a high-risk device
    if len(high_risk) > 0:
        fig1 = plot_sensor_trends(df, high_risk[0])
        fig1.savefig('sensor_trends.png', dpi=150)
        print("Saved: sensor_trends.png")
    
    # Overall dashboard
    fig2 = plot_risk_distribution(df)
    fig2.savefig('risk_dashboard.png', dpi=150)
    print("Saved: risk_dashboard.png")
    
    # Step 6: Business summary
    print("\n6. Generating business impact report...")
    summary = create_executive_summary(acc, prec, rec, df['device_id'].nunique())
    print(summary)
    
    # Save summary
    with open('executive_summary.txt', 'w') as f:
        f.write(summary)
    
    # Save high-risk device list
    risk_report = df[df['device_id'].isin(high_risk)].groupby('device_id').agg({
        'health_score': 'mean',
        'temperature': 'mean',
        'vibration': 'mean'
    }).round(2)
    
    risk_report.to_csv('high_risk_devices.csv')
    print("\nSaved: high_risk_devices.csv")
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()