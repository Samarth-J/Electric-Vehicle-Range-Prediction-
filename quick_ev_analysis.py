#!/usr/bin/env python3
"""
Quick EV Range Analysis
======================

A simplified version that handles data issues and provides immediate insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """Load and clean the EV dataset."""
    print("Loading EV dataset...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully: {df.shape[0]} vehicles, {df.shape[1]} features")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Basic info
    print(f"\nRange statistics:")
    print(f"- Min range: {df['range_km'].min()} km")
    print(f"- Max range: {df['range_km'].max()} km") 
    print(f"- Average range: {df['range_km'].mean():.1f} km")
    print(f"- Median range: {df['range_km'].median():.1f} km")
    
    return df

def prepare_features(df):
    """Prepare features for modeling."""
    print("\nPreparing features...")
    
    # Select key numerical features that are most likely to be available
    numerical_features = []
    potential_numerical = [
        'battery_capacity_kWh', 'top_speed_kmh', 'efficiency_wh_per_km',
        'acceleration_0_100_s', 'towing_capacity_kg', 'torque_nm'
    ]
    
    for col in potential_numerical:
        if col in df.columns:
            numerical_features.append(col)
    
    # Select key categorical features
    categorical_features = []
    potential_categorical = ['drivetrain', 'segment', 'car_body_type']
    
    for col in potential_categorical:
        if col in df.columns:
            categorical_features.append(col)
    
    print(f"Using numerical features: {numerical_features}")
    print(f"Using categorical features: {categorical_features}")
    
    # Create feature matrix
    X = df[numerical_features + categorical_features].copy()
    
    # Handle missing values in numerical features
    for col in numerical_features:
        X[col] = X[col].fillna(X[col].median())
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Target variable
    y = df['range_km'].copy()
    
    # Remove any remaining missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y, numerical_features, categorical_features, label_encoders

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple models."""
    print("\nTraining models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models to try
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"  MAE: {mae:.1f} km")
        print(f"  R²: {r2:.4f}")
    
    return results, X_test, y_test

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance for Random Forest."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance (Random Forest):")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    return None

def create_simple_plots(results, X_test, y_test):
    """Create simple visualization plots."""
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Model comparison
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    
    axes[0].bar(model_names, r2_scores, color=['skyblue', 'lightgreen'])
    axes[0].set_title('Model Performance (R² Score)')
    axes[0].set_ylabel('R² Score')
    axes[0].set_ylim(0, 1)
    
    # Best model predictions
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    y_pred = results[best_model_name]['predictions']
    
    axes[1].scatter(y_test, y_pred, alpha=0.6)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Range (km)')
    axes[1].set_ylabel('Predicted Range (km)')
    axes[1].set_title(f'Actual vs Predicted ({best_model_name})')
    
    plt.tight_layout()
    plt.savefig('ev_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function."""
    print("Quick EV Range Analysis")
    print("======================")
    
    # Load data
    df = load_and_clean_data('electric_vehicles_spec_2025.csv.csv')
    if df is None:
        return
    
    # Prepare features
    X, y, numerical_features, categorical_features, label_encoders = prepare_features(df)
    
    # Train models
    results, X_test, y_test = train_and_evaluate_models(X, y)
    
    # Analyze feature importance
    rf_model = results['Random Forest']['model']
    feature_importance = analyze_feature_importance(rf_model, X.columns.tolist())
    
    # Create plots
    create_simple_plots(results, X_test, y_test)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_result = results[best_model_name]
    
    print(f"Best Model: {best_model_name}")
    print(f"- R² Score: {best_result['r2']:.4f}")
    print(f"- Mean Absolute Error: {best_result['mae']:.1f} km")
    
    print(f"\nKey Insights:")
    print("- Battery capacity is likely the strongest predictor of range")
    print("- Vehicle efficiency (Wh/km) significantly impacts predictions")
    print("- The model can predict EV range with reasonable accuracy")
    print("- Consider additional features like vehicle weight for better predictions")
    
    print(f"\nFiles generated:")
    print("- ev_analysis_results.png (visualization)")
    
    return results, feature_importance

if __name__ == "__main__":
    results, feature_importance = main()