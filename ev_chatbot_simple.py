#!/usr/bin/env python3
"""
Simple Electric Vehicle Chatbot
================================

A streamlined chatbot for exploring electric vehicles.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("🔄 Loading EV database...")
df = pd.read_csv('electric_vehicles_spec_2025.csv.csv')
print(f"✅ Loaded {len(df)} vehicles from {df['brand'].nunique()} brands\n")

# Train model
print("🤖 Training prediction model...")
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
print("✅ Model ready!\n")

print("=" * 70)
print("🚗 ELECTRIC VEHICLE CHATBOT 🔋")
print("=" * 70)
print("\n👋 Welcome! Ask me about electric vehicles!")
print("\nTry these commands:")
print("  • 'stats' - Database statistics")
print("  • 'top' - Top 5 EVs by range")
print("  • 'tesla' - Show Tesla vehicles")
print("  • 'recommend' - Get recommendations")
print("  • 'help' - Show all commands")
print("  • 'quit' - Exit\n")

while True:
    try:
        user_input = input("You: ").strip().lower()
        
        if not user_input:
            continue
        
        if user_input in ['quit', 'exit', 'bye']:
            print("\n👋 Goodbye! Happy EV shopping! 🚗⚡\n")
            break
        
        print()  # Blank line
        
        # Handle queries
        if 'stats' in user_input or 'overview' in user_input:
            print("📊 EV Database Statistics:")
            print(f"  • Total vehicles: {len(df)}")
            print(f"  • Brands: {df['brand'].nunique()}")
            print(f"  • Average range: {df['range_km'].mean():.1f} km")
            print(f"  • Max range: {df['range_km'].max()} km")
            print(f"  • Min range: {df['range_km'].min()} km")
            print(f"  • Average battery: {df['battery_capacity_kWh'].mean():.1f} kWh")
        
        elif 'top' in user_input or 'best' in user_input:
            print("🏆 Top 5 EVs by Range:")
            top = df.nlargest(5, 'range_km')
            for i, (idx, row) in enumerate(top.iterrows(), 1):
                print(f"  {i}. {row['brand']} {row['model']}: {row['range_km']} km ({row['battery_capacity_kWh']} kWh)")
        
        elif 'tesla' in user_input:
            tesla = df[df['brand'].str.lower().str.contains('tesla', na=False)]
            if len(tesla) > 0:
                print(f"🚗 Tesla Vehicles ({len(tesla)} found):")
                for idx, row in tesla.head(10).iterrows():
                    print(f"  • {row['model']}: {row['range_km']} km, {row['battery_capacity_kWh']} kWh")
            else:
                print("❌ No Tesla vehicles found")
        
        elif 'bmw' in user_input:
            bmw = df[df['brand'].str.lower().str.contains('bmw', na=False)]
            if len(bmw) > 0:
                print(f"🚗 BMW Vehicles ({len(bmw)} found):")
                for idx, row in bmw.head(10).iterrows():
                    print(f"  • {row['model']}: {row['range_km']} km, {row['battery_capacity_kWh']} kWh")
            else:
                print("❌ No BMW vehicles found")
        
        elif 'audi' in user_input:
            audi = df[df['brand'].str.lower().str.contains('audi', na=False)]
            if len(audi) > 0:
                print(f"🚗 Audi Vehicles ({len(audi)} found):")
                for idx, row in audi.head(10).iterrows():
                    print(f"  • {row['model']}: {row['range_km']} km, {row['battery_capacity_kWh']} kWh")
            else:
                print("❌ No Audi vehicles found")
        
        elif 'recommend' in user_input or 'suggest' in user_input:
            if 'long' in user_input:
                recs = df[df['range_km'] >= 400].nlargest(5, 'range_km')
                print("✨ Recommended Long-Range EVs (400+ km):")
                for idx, row in recs.iterrows():
                    print(f"  • {row['brand']} {row['model']}: {row['range_km']} km")
            else:
                recs = df.nlargest(5, 'range_km')
                print("✨ Top Recommended EVs:")
                for idx, row in recs.iterrows():
                    print(f"  • {row['brand']} {row['model']}: {row['range_km']} km, {row['drivetrain']}")
        
        elif 'average' in user_input:
            if 'range' in user_input:
                print(f"📊 Average EV range: {df['range_km'].mean():.1f} km")
            elif 'battery' in user_input:
                print(f"🔋 Average battery capacity: {df['battery_capacity_kWh'].mean():.1f} kWh")
            else:
                print(f"📊 Average range: {df['range_km'].mean():.1f} km")
                print(f"🔋 Average battery: {df['battery_capacity_kWh'].mean():.1f} kWh")
        
        elif 'how many' in user_input:
            if 'vehicle' in user_input or 'car' in user_input:
                print(f"📊 Database contains {len(df)} electric vehicles")
            elif 'brand' in user_input:
                print(f"📊 Database has {df['brand'].nunique()} different brands")
            else:
                print(f"📊 {len(df)} vehicles from {df['brand'].nunique()} brands")
        
        elif 'help' in user_input:
            print("🤖 Available Commands:")
            print("  • 'stats' - Show database statistics")
            print("  • 'top' - Top 5 EVs by range")
            print("  • 'tesla' / 'bmw' / 'audi' - Show brand vehicles")
            print("  • 'recommend' - Get recommendations")
            print("  • 'recommend long' - Long-range EVs")
            print("  • 'average range' - Average statistics")
            print("  • 'how many vehicles' - Count vehicles")
            print("  • 'help' - Show this help")
            print("  • 'quit' - Exit chatbot")
        
        else:
            print("🤔 I'm not sure I understand. Try:")
            print("  • 'stats' for overview")
            print("  • 'top' for best EVs")
            print("  • 'tesla' for Tesla vehicles")
            print("  • 'help' for all commands")
        
        print()  # Blank line
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Happy EV shopping! 🚗⚡\n")
        break
    except Exception as e:
        print(f"❌ Error: {str(e)}\n")

print("Thanks for using the EV Chatbot!")