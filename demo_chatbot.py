#!/usr/bin/env python3
"""
EV Chatbot Demo
===============

A demonstration of the EV chatbot with pre-programmed queries.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_and_train():
    """Load data and train model."""
    print("🔄 Loading EV database and training model...\n")
    
    df = pd.read_csv('electric_vehicles_spec_2025.csv.csv')
    
    # Prepare features
    numerical_features = ['battery_capacity_kWh', 'top_speed_kmh', 'efficiency_wh_per_km',
                         'acceleration_0_100_s', 'towing_capacity_kg', 'torque_nm']
    categorical_features = ['drivetrain', 'segment', 'car_body_type']
    
    existing_numerical = [col for col in numerical_features if col in df.columns]
    existing_categorical = [col for col in categorical_features if col in df.columns]
    
    X = df[existing_numerical + existing_categorical].copy()
    
    for col in existing_numerical:
        X[col] = X[col].fillna(X[col].median())
    
    label_encoders = {}
    for col in existing_categorical:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    y = df['range_km'].copy()
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("✅ Model trained successfully!")
    print(f"📊 Database loaded: {len(df)} vehicles\n")
    
    return df, model

def demo_queries(df):
    """Run demonstration queries."""
    print("=" * 70)
    print("🚗 ELECTRIC VEHICLE CHATBOT DEMO 🔋")
    print("=" * 70)
    print("\nDemonstrating chatbot capabilities with sample queries...\n")
    
    queries = [
        ("📊 Database Statistics", "stats"),
        ("🏆 Top 5 EVs by Range", "top_range"),
        ("🔍 Tesla Vehicles", "tesla"),
        ("🔋 Best Battery Capacity", "best_battery"),
        ("✨ Long Range Recommendations", "recommend_long"),
        ("❓ Average Range", "average")
    ]
    
    for title, query_type in queries:
        print("─" * 70)
        print(f"\n{title}")
        print("─" * 70)
        
        if query_type == "stats":
            print(f"Total vehicles: {len(df)}")
            print(f"Number of brands: {df['brand'].nunique()}")
            print(f"Average range: {df['range_km'].mean():.1f} km")
            print(f"Maximum range: {df['range_km'].max()} km")
            print(f"Minimum range: {df['range_km'].min()} km")
            print(f"Average battery: {df['battery_capacity_kWh'].mean():.1f} kWh")
        
        elif query_type == "top_range":
            top = df.nlargest(5, 'range_km')[['brand', 'model', 'range_km', 'battery_capacity_kWh']]
            for idx, row in top.iterrows():
                print(f"  {row['brand']} {row['model']}: {row['range_km']} km ({row['battery_capacity_kWh']} kWh)")
        
        elif query_type == "tesla":
            tesla = df[df['brand'].str.lower().str.contains('tesla', na=False)]
            if len(tesla) > 0:
                print(f"Found {len(tesla)} Tesla vehicles:")
                for idx, row in tesla.head(5).iterrows():
                    print(f"  - {row['model']}: {row['range_km']} km, {row['battery_capacity_kWh']} kWh")
            else:
                print("No Tesla vehicles found in database")
        
        elif query_type == "best_battery":
            top = df.nlargest(5, 'battery_capacity_kWh')[['brand', 'model', 'battery_capacity_kWh', 'range_km']]
            for idx, row in top.iterrows():
                print(f"  {row['brand']} {row['model']}: {row['battery_capacity_kWh']} kWh ({row['range_km']} km)")
        
        elif query_type == "recommend_long":
            recs = df[df['range_km'] >= 400].nlargest(5, 'range_km')[['brand', 'model', 'range_km']]
            print("Long-range EVs (400+ km):")
            for idx, row in recs.iterrows():
                print(f"  - {row['brand']} {row['model']}: {row['range_km']} km")
        
        elif query_type == "average":
            avg_range = df['range_km'].mean()
            avg_battery = df['battery_capacity_kWh'].mean()
            print(f"Average EV range: {avg_range:.1f} km")
            print(f"Average battery capacity: {avg_battery:.1f} kWh")
        
        print()
    
    print("=" * 70)
    print("\n✅ Demo complete!")
    print("\nTo use the interactive chatbot, run:")
    print("  python ev_chatbot.py")
    print("\nFor the web interface, run:")
    print("  streamlit run ev_chatbot_web.py")
    print("\n" + "=" * 70)

def main():
    """Main demo function."""
    df, model = load_and_train()
    demo_queries(df)

if __name__ == "__main__":
    main()