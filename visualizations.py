# visualizations.py
# Dashboard visualizations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_sensor_trends(df, device_id=None):
    """
    Plot sensor data over time
    """
    if device_id is None:
        device_id = df['device_id'].iloc[0]
    
    device_data = df[df['device_id'] == device_id].tail(24*7)  # Last week
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Temperature
    axes[0].plot(device_data['timestamp'], device_data['temperature'], 'b-', alpha=0.7)
    axes[0].plot(device_data['timestamp'], device_data['temperature_24h_mean'], 'r--', alpha=0.5)
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title(f'Device {device_id} - Sensor Readings')
    axes[0].grid(True, alpha=0.3)
    
    # Vibration
    axes[1].plot(device_data['timestamp'], device_data['vibration'], 'g-', alpha=0.7)
    axes[1].set_ylabel('Vibration')
    axes[1].grid(True, alpha=0.3)
    
    # Health Score
    axes[2].plot(device_data['timestamp'], device_data['health_score'], 'purple', alpha=0.7)
    axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Risk Threshold')
    axes[2].set_ylabel('Health Score')
    axes[2].set_xlabel('Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_risk_distribution(df):
    """
    Show risk across all devices
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Health score distribution
    axes[0, 0].hist(df.groupby('device_id')['health_score'].mean(), bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Average Health Score')
    axes[0, 0].set_ylabel('Number of Devices')
    axes[0, 0].set_title('Health Score Distribution')
    
    # Risk by equipment type
    risk_by_type = df.groupby('type')['high_risk'].mean()
    axes[0, 1].bar(risk_by_type.index, risk_by_type.values, color=['green', 'yellow', 'orange', 'red'])
    axes[0, 1].set_xlabel('Equipment Type')
    axes[0, 1].set_ylabel('Risk Rate')
    axes[0, 1].set_title('Risk by Equipment Type')
    
    # Temperature anomalies over time
    daily_anomalies = df.groupby(df['timestamp'].dt.date)['temp_anomaly'].mean()
    axes[1, 0].plot(daily_anomalies.index, daily_anomalies.values, 'r-', marker='o')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Anomaly Rate')
    axes[1, 0].set_title('Temperature Anomalies Trend')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Top 10 risky devices
    worst_devices = df.groupby('device_id')['health_score'].mean().sort_values().head(10)
    axes[1, 1].barh(range(len(worst_devices)), worst_devices.values, color='red')
    axes[1, 1].set_yticks(range(len(worst_devices)))
    axes[1, 1].set_yticklabels(worst_devices.index)
    axes[1, 1].set_xlabel('Health Score')
    axes[1, 1].set_title('Top 10 High-Risk Devices')
    
    plt.suptitle('Predictive Maintenance Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_executive_summary(accuracy, precision, recall, n_devices):
    """
    Simple text summary for management
    """
    # Calculate savings (simplified)
    prevented_failures = int(n_devices * 0.1 * precision)  # 10% failure rate
    savings = prevented_failures * 50000  # $50k per prevented failure
    
    summary = f"""
    PREDICTIVE MAINTENANCE - EXECUTIVE SUMMARY
    ==========================================
    
    Model Performance:
    - Accuracy: {accuracy:.1%}
    - Precision: {precision:.1%}
    - Recall: {recall:.1%}
    
    Business Impact:
    - Devices Monitored: {n_devices}
    - Prevented Failures: ~{prevented_failures}
    - Estimated Annual Savings: ${savings:,}
    - ROI: {(savings / 100000 - 1) * 100:.0f}%  # Assuming $100k implementation cost
    
    Recommendation: Deploy to production with phased rollout
    """
    
    return summary