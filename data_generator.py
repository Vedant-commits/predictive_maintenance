# data_generator.py
# Started: [Today's date] - Initial sensor data simulation for Cardinal Health equipment

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_sensor_data():
    """
    Simulating sensor readings from medical equipment
    Based on my experience with Cardinal Health supply chain data
    """
    # Using 42 for reproducibility during testing
    np.random.seed(42)
    
    # Equipment we actually use at hospitals
    equipment = ['MRI', 'X-Ray', 'Ventilator', 'Infusion Pump']  
    
    data = []
    n_devices = 500  # Start with 500, can scale later
    
    print("Generating device data...")
    
    for i in range(n_devices):
        device_type = np.random.choice(equipment)
        device_id = f'DEV{i:04d}'  # Simple ID format
        
        # Each equipment has different normal ranges (from domain knowledge)
        if device_type == 'MRI':
            temp_baseline = 18  # MRIs run cold
            vib_baseline = 2.5  # Higher vibration normal
        elif device_type == 'Ventilator':
            temp_baseline = 25  # Run warmer
            vib_baseline = 0.8  # Low vibration
        else:
            temp_baseline = 22
            vib_baseline = 1.5
            
        # Simulate 90 days of hourly data (keeping it manageable)
        for hour in range(90 * 24):
            # Add some realistic patterns
            daily_pattern = np.sin(2 * np.pi * hour / 24) * 2
            weekly_pattern = np.sin(2 * np.pi * hour / (24 * 7)) * 1
            
            temp = temp_baseline + daily_pattern + np.random.normal(0, 1)
            vib = vib_baseline + np.random.normal(0, 0.3)
            
            # Sometimes equipment degrades (about 10% of devices)
            if i < n_devices * 0.1 and hour > 1800:  # After 75 days
                degradation = (hour - 1800) / 500
                temp += degradation * 5
                vib += degradation * 2
            
            timestamp = datetime.now() - timedelta(hours=(90*24 - hour))
            
            data.append({
                'device_id': device_id,
                'type': device_type,
                'timestamp': timestamp,
                'temperature': round(temp, 2),
                'vibration': round(vib, 3),
                'pressure': round(100 + np.random.normal(0, 10), 1),
                'usage_hours': hour
            })
    
    df = pd.DataFrame(data)
    print(f"Created {len(df)} records")
    return df

if __name__ == "__main__":
    # Test the data generation
    test_data = create_sensor_data()
    print(test_data.head())
    print(f"\nData shape: {test_data.shape}")
    print(f"Device types: {test_data['type'].value_counts().to_dict()}")