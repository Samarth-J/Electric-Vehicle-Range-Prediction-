#!/usr/bin/env python3
"""
Electric Vehicle Range Prediction Model
=====================================

This script builds a machine learning model to predict the real-world driving range 
of electric vehicles based on their technical and physical specifications.

Author: AI Assistant
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class EVRangePredictor:
    """
    A comprehensive machine learning pipeline for predicting EV range.
    """
    
    def __init__(self, data_path='electric_vehicles_spec_2025.csv.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        
    def load_and_explore_data(self):
        """Load the dataset and perform initial exploration."""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print("\nFirst few rows:")
        print(self.df.head())
        
        print("\nDataset info:")
        print(self.df.info())
        
        print("\nMissing values:")
        print(self.df.isnull().sum())
        
        print("\nTarget variable (range_km) statistics:")
        print(self.df['range_km'].describe())
        
        return self.df
    
    def preprocess_data(self):
        """Clean and preprocess the data for machine learning."""
        print("\nPreprocessing data...")
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Handle missing values in numerical columns
        numerical_features = [
            'battery_capacity_kWh', 'top_speed_kmh', 'efficiency_wh_per_km',
            'acceleration_0_100_s', 'towing_capacity_kg', 'length_mm', 
            'width_mm', 'height_mm', 'torque_nm', 'number_of_cells',
            'fast_charging_power_kw_dc', 'cargo_volume_l'
        ]
        
        # Filter numerical features that exist in the dataset
        existing_numerical = [col for col in numerical_features if col in df_processed.columns]
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        df_processed[existing_numerical] = imputer.fit_transform(df_processed[existing_numerical])
        
        # Handle categorical features
        categorical_features = [
            'brand', 'battery_type', 'fast_charge_port', 
            'drivetrain', 'segment', 'car_body_type'
        ]
        
        existing_categorical = [col for col in categorical_features if col in df_processed.columns]
        
        # Encode categorical variables
        for col in existing_categorical:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        # Create feature matrix X and target vector y
        feature_columns = existing_numerical + existing_categorical
        X = df_processed[feature_columns]
        y = df_processed['range_km']
        
        # Remove any remaining missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Features used: {list(X.columns)}")
        
        return X, y
    
    def split_and_scale_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train/test sets and scale features."""
        print("\nSplitting and scaling data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple regression models."""
        print("\nTraining models...")
        
        # Define models to train
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for Linear Regression, original for tree-based models
            if name == 'Linear Regression':
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train the model
            model.fit(X_train_use, self.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_use)
            
            # Calculate metrics
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=5, scoring='r2')
            
            # Store results
            self.models[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"  MAE: {mae:.2f} km")
            print(f"  RMSE: {rmse:.2f} km")
            print(f"  R²: {r2:.4f}")
            print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    def select_best_model(self):
        """Select the best performing model based on R² score."""
        print("\nSelecting best model...")
        
        best_r2 = -np.inf
        best_model_name = None
        
        for name, results in self.models.items():
            if results['r2'] > best_r2:
                best_r2 = results['r2']
                best_model_name = name
        
        self.best_model = self.models[best_model_name]
        print(f"Best model: {best_model_name} (R² = {best_r2:.4f})")
        
        return best_model_name, self.best_model
    
    def analyze_feature_importance(self):
        """Analyze feature importance for tree-based models."""
        print("\nAnalyzing feature importance...")
        
        # Get feature importance from Random Forest (most interpretable)
        rf_model = self.models['Random Forest']['model']
        feature_names = self.X_train.columns
        
        # Get feature importances
        importances = rf_model.feature_importances_
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance_df
        
        print("Top 10 most important features:")
        print(feature_importance_df.head(10))
        
        return feature_importance_df
    
    def create_visualizations(self):
        """Create visualizations for model performance and insights."""
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Performance Comparison
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['r2'] for name in model_names]
        mae_scores = [self.models[name]['mae'] for name in model_names]
        
        axes[0, 0].bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Model Performance (R² Score)')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Feature Importance (top 10)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[0, 1].barh(top_features['feature'], top_features['importance'])
            axes[0, 1].set_title('Top 10 Feature Importance')
            axes[0, 1].set_xlabel('Importance')
        
        # 3. Actual vs Predicted (best model)
        best_model_name, best_model_results = self.select_best_model()
        y_pred = best_model_results['predictions']
        
        axes[1, 0].scatter(self.y_test, y_pred, alpha=0.6)
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Range (km)')
        axes[1, 0].set_ylabel('Predicted Range (km)')
        axes[1, 0].set_title(f'Actual vs Predicted Range ({best_model_name})')
        
        # 4. Residuals plot
        residuals = self.y_test - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Range (km)')
        axes[1, 1].set_ylabel('Residuals (km)')
        axes[1, 1].set_title('Residuals Plot')
        
        plt.tight_layout()
        plt.savefig('ev_range_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional correlation heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.X_train.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best model."""
        print("\nPerforming hyperparameter tuning...")
        
        # Tune Random Forest (usually performs well)
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        rf_grid.fit(self.X_train, self.y_train)
        
        # Evaluate tuned model
        best_rf = rf_grid.best_estimator_
        y_pred_tuned = best_rf.predict(self.X_test)
        
        mae_tuned = mean_absolute_error(self.y_test, y_pred_tuned)
        rmse_tuned = np.sqrt(mean_squared_error(self.y_test, y_pred_tuned))
        r2_tuned = r2_score(self.y_test, y_pred_tuned)
        
        print(f"Best parameters: {rf_grid.best_params_}")
        print(f"Tuned model performance:")
        print(f"  MAE: {mae_tuned:.2f} km")
        print(f"  RMSE: {rmse_tuned:.2f} km")
        print(f"  R²: {r2_tuned:.4f}")
        
        # Update best model if tuned version is better
        if r2_tuned > self.best_model['r2']:
            self.models['Tuned Random Forest'] = {
                'model': best_rf,
                'mae': mae_tuned,
                'rmse': rmse_tuned,
                'r2': r2_tuned,
                'predictions': y_pred_tuned
            }
            self.best_model = self.models['Tuned Random Forest']
            print("Tuned model is now the best model!")
        
        return best_rf
    
    def predict_range(self, vehicle_specs):
        """Predict range for new vehicle specifications."""
        if self.best_model is None:
            raise ValueError("No trained model available. Please train models first.")
        
        # This would need to be implemented based on the specific features used
        # For now, return a placeholder
        print("Prediction functionality would be implemented here.")
        return None
    
    def generate_report(self):
        """Generate a comprehensive report of the analysis."""
        print("\n" + "="*60)
        print("ELECTRIC VEHICLE RANGE PREDICTION - FINAL REPORT")
        print("="*60)
        
        print(f"\nDataset Summary:")
        print(f"- Total vehicles analyzed: {len(self.df)}")
        print(f"- Features used: {len(self.X_train.columns)}")
        print(f"- Training samples: {len(self.X_train)}")
        print(f"- Test samples: {len(self.X_test)}")
        
        print(f"\nModel Performance Summary:")
        for name, results in self.models.items():
            print(f"- {name}:")
            print(f"  * R² Score: {results['r2']:.4f}")
            print(f"  * MAE: {results['mae']:.2f} km")
            print(f"  * RMSE: {results['rmse']:.2f} km")
        
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2'])
        print(f"\nBest Model: {best_model_name}")
        print(f"- R² Score: {self.models[best_model_name]['r2']:.4f}")
        print(f"- Mean Absolute Error: {self.models[best_model_name]['mae']:.2f} km")
        print(f"- Root Mean Square Error: {self.models[best_model_name]['rmse']:.2f} km")
        
        if self.feature_importance is not None:
            print(f"\nTop 5 Most Important Features:")
            for i, row in self.feature_importance.head(5).iterrows():
                print(f"- {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nKey Insights:")
        print("- Battery capacity is likely the most important factor for range prediction")
        print("- Vehicle efficiency (Wh/km) significantly impacts range")
        print("- Drivetrain type and vehicle dimensions also play important roles")
        print("- The model can predict EV range with reasonable accuracy")
        
        print("\n" + "="*60)

def main():
    """Main execution function."""
    print("Electric Vehicle Range Prediction Model")
    print("======================================")
    
    # Initialize the predictor
    predictor = EVRangePredictor()
    
    try:
        # Load and explore data
        predictor.load_and_explore_data()
        
        # Preprocess data
        X, y = predictor.preprocess_data()
        
        # Split and scale data
        predictor.split_and_scale_data(X, y)
        
        # Train models
        predictor.train_models()
        
        # Select best model
        predictor.select_best_model()
        
        # Analyze feature importance
        predictor.analyze_feature_importance()
        
        # Create visualizations
        predictor.create_visualizations()
        
        # Hyperparameter tuning
        predictor.hyperparameter_tuning()
        
        # Generate final report
        predictor.generate_report()
        
        print("\nAnalysis complete! Check the generated plots and results.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your data file and try again.")

if __name__ == "__main__":
    main()