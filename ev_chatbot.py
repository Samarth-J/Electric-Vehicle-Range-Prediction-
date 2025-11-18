#!/usr/bin/env python3
"""
Electric Vehicle Chatbot
========================

An interactive chatbot that provides information about electric vehicles,
predicts range, and answers questions about the EV dataset.

Author: AI Assistant
Date: November 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class EVChatbot:
    """
    Interactive chatbot for electric vehicle information and predictions.
    """
    
    def __init__(self, data_path='electric_vehicles_spec_2025.csv.csv'):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.load_data_and_train()
        
    def load_data_and_train(self):
        """Load data and train the prediction model."""
        print("🔄 Loading EV database and training model...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Prepare features
        numerical_features = [
            'battery_capacity_kWh', 'top_speed_kmh', 'efficiency_wh_per_km',
            'acceleration_0_100_s', 'towing_capacity_kg', 'torque_nm'
        ]
        
        categorical_features = ['drivetrain', 'segment', 'car_body_type']
        
        # Filter existing features
        existing_numerical = [col for col in numerical_features if col in self.df.columns]
        existing_categorical = [col for col in categorical_features if col in self.df.columns]
        
        # Create feature matrix
        X = self.df[existing_numerical + existing_categorical].copy()
        
        # Handle missing values
        for col in existing_numerical:
            X[col] = X[col].fillna(X[col].median())
        
        # Encode categorical features
        for col in existing_categorical:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        y = self.df['range_km'].copy()
        
        # Remove missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        self.feature_columns = X.columns.tolist()
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        print("✅ Model trained successfully!")
        print(f"📊 Database loaded: {len(self.df)} vehicles\n")
    
    def get_stats(self):
        """Get general statistics about the dataset."""
        stats = {
            'total_vehicles': len(self.df),
            'brands': self.df['brand'].nunique(),
            'avg_range': self.df['range_km'].mean(),
            'max_range': self.df['range_km'].max(),
            'min_range': self.df['range_km'].min(),
            'avg_battery': self.df['battery_capacity_kWh'].mean(),
            'drivetrain_types': self.df['drivetrain'].value_counts().to_dict()
        }
        return stats
    
    def search_vehicles(self, query):
        """Search for vehicles by brand or model."""
        query_lower = query.lower()
        results = self.df[
            self.df['brand'].str.lower().str.contains(query_lower, na=False) |
            self.df['model'].str.lower().str.contains(query_lower, na=False)
        ]
        return results
    
    def get_top_vehicles(self, criteria='range_km', n=5):
        """Get top vehicles by a specific criteria."""
        if criteria in self.df.columns:
            top = self.df.nlargest(n, criteria)[['brand', 'model', criteria, 'range_km', 'battery_capacity_kWh']]
            return top
        return None
    
    def get_recommendations(self, min_range=None, drivetrain=None):
        """Get vehicle recommendations based on criteria."""
        filtered = self.df.copy()
        
        if min_range:
            filtered = filtered[filtered['range_km'] >= min_range]
        
        if drivetrain:
            filtered = filtered[filtered['drivetrain'].str.lower() == drivetrain.lower()]
        
        return filtered[['brand', 'model', 'range_km', 'battery_capacity_kWh', 'drivetrain', 'segment']].head(10)
    
    def handle_query(self, user_input):
        """Process user queries and provide appropriate responses."""
        user_input_lower = user_input.lower()
        
        # Greetings
        if any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return "👋 Hello! I'm your EV assistant. I can help you with:\n" \
                   "- Vehicle information and comparisons\n" \
                   "- Range predictions\n" \
                   "- Top EVs by various criteria\n" \
                   "- Statistics about the EV market\n" \
                   "- Recommendations based on your needs\n\n" \
                   "What would you like to know?"
        
        # Statistics
        elif any(word in user_input_lower for word in ['stats', 'statistics', 'overview', 'summary']):
            stats = self.get_stats()
            return f"📊 **EV Database Statistics:**\n" \
                   f"- Total vehicles: {stats['total_vehicles']}\n" \
                   f"- Number of brands: {stats['brands']}\n" \
                   f"- Average range: {stats['avg_range']:.1f} km\n" \
                   f"- Maximum range: {stats['max_range']} km\n" \
                   f"- Minimum range: {stats['min_range']} km\n" \
                   f"- Average battery capacity: {stats['avg_battery']:.1f} kWh"
        
        # Top vehicles
        elif 'top' in user_input_lower or 'best' in user_input_lower:
            if 'range' in user_input_lower:
                top = self.get_top_vehicles('range_km', 5)
                response = "🏆 **Top 5 EVs by Range:**\n\n"
                for idx, row in top.iterrows():
                    response += f"{row['brand']} {row['model']}: {row['range_km']} km\n"
                return response
            elif 'battery' in user_input_lower:
                top = self.get_top_vehicles('battery_capacity_kWh', 5)
                response = "🔋 **Top 5 EVs by Battery Capacity:**\n\n"
                for idx, row in top.iterrows():
                    response += f"{row['brand']} {row['model']}: {row['battery_capacity_kWh']} kWh\n"
                return response
            else:
                top = self.get_top_vehicles('range_km', 5)
                response = "🏆 **Top 5 EVs Overall (by Range):**\n\n"
                for idx, row in top.iterrows():
                    response += f"{row['brand']} {row['model']}: {row['range_km']} km\n"
                return response
        
        # Search for specific brand
        elif any(brand in user_input_lower for brand in ['tesla', 'bmw', 'audi', 'ford', 'hyundai', 'kia', 'byd']):
            brands = ['tesla', 'bmw', 'audi', 'ford', 'hyundai', 'kia', 'byd', 'mercedes', 'volkswagen']
            brand = next((b for b in brands if b in user_input_lower), None)
            
            if brand:
                results = self.search_vehicles(brand)
                if len(results) > 0:
                    response = f"🚗 **{brand.upper()} Electric Vehicles ({len(results)} found):**\n\n"
                    for idx, row in results.head(10).iterrows():
                        response += f"- {row['model']}: {row['range_km']} km range, {row['battery_capacity_kWh']} kWh battery\n"
                    return response
                else:
                    return f"❌ No vehicles found for {brand.upper()}"
        
        # Recommendations
        elif 'recommend' in user_input_lower or 'suggest' in user_input_lower:
            if 'long range' in user_input_lower or 'longest' in user_input_lower:
                recs = self.get_recommendations(min_range=400)
                response = "✨ **Recommended Long-Range EVs (400+ km):**\n\n"
                for idx, row in recs.head(5).iterrows():
                    response += f"- {row['brand']} {row['model']}: {row['range_km']} km\n"
                return response
            else:
                recs = self.get_recommendations()
                response = "✨ **Top Recommended EVs:**\n\n"
                for idx, row in recs.head(5).iterrows():
                    response += f"- {row['brand']} {row['model']}: {row['range_km']} km, {row['drivetrain']}\n"
                return response
        
        # Help
        elif 'help' in user_input_lower or 'what can you do' in user_input_lower:
            return "🤖 **I can help you with:**\n\n" \
                   "1. 📊 Get statistics: 'show stats' or 'overview'\n" \
                   "2. 🏆 Find top EVs: 'top range' or 'best battery'\n" \
                   "3. 🔍 Search vehicles: 'show me Tesla' or 'BMW models'\n" \
                   "4. ✨ Get recommendations: 'recommend long range EVs'\n" \
                   "5. ❓ Ask questions: 'what is the average range?'\n\n" \
                   "Just ask me anything about electric vehicles!"
        
        # Average questions
        elif 'average' in user_input_lower:
            if 'range' in user_input_lower:
                avg = self.df['range_km'].mean()
                return f"📊 The average EV range in our database is **{avg:.1f} km**"
            elif 'battery' in user_input_lower:
                avg = self.df['battery_capacity_kWh'].mean()
                return f"🔋 The average battery capacity is **{avg:.1f} kWh**"
        
        # Count questions
        elif 'how many' in user_input_lower:
            if 'vehicle' in user_input_lower or 'car' in user_input_lower:
                return f"📊 Our database contains **{len(self.df)} electric vehicles**"
            elif 'brand' in user_input_lower:
                return f"📊 We have **{self.df['brand'].nunique()} different brands** in our database"
        
        # Default response
        else:
            return "🤔 I'm not sure I understand. Try asking:\n" \
                   "- 'show stats' for database overview\n" \
                   "- 'top range' for best EVs\n" \
                   "- 'show me [brand]' for specific vehicles\n" \
                   "- 'help' for more options"
    
    def run(self):
        """Run the interactive chatbot."""
        print("=" * 60)
        print("🚗 ELECTRIC VEHICLE CHATBOT 🔋")
        print("=" * 60)
        print("\n👋 Welcome! I'm your EV assistant.")
        print("I can help you explore electric vehicles, predict range, and more!")
        print("\nType 'help' for available commands or 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n👋 Goodbye! Happy EV shopping! 🚗⚡")
                    break
                
                response = self.handle_query(user_input)
                print(f"\n🤖 Bot: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Happy EV shopping! 🚗⚡")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")

def main():
    """Main function to run the chatbot."""
    chatbot = EVChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()