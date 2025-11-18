# Electric Vehicle Chatbot Guide 🚗🤖

## Overview

The EV Chatbot is an AI-powered assistant that helps users explore electric vehicles, get recommendations, compare models, and predict range based on specifications.

## 🎯 Features

### 1. **Interactive Q&A**
- Ask questions about electric vehicles
- Get statistics about the EV market
- Search for specific brands and models
- Compare different vehicles

### 2. **Smart Recommendations**
- Find EVs based on range requirements
- Get suggestions for long-range vehicles
- Filter by drivetrain type (FWD/RWD/AWD)
- Discover top-rated vehicles

### 3. **Range Prediction**
- Predict EV range based on specifications
- Input battery capacity, efficiency, and other specs
- Get accurate predictions using ML model (95.77% accuracy)

### 4. **Data Analytics**
- View market statistics
- Analyze trends across brands
- Compare battery capacities and ranges
- Explore vehicle segments

## 📦 Available Versions

### 1. Command-Line Chatbot (`ev_chatbot.py`)
**Best for:** Quick queries and terminal users

**Run:**
```bash
python ev_chatbot.py
```

**Features:**
- Interactive text-based interface
- Real-time responses
- Simple and fast
- No additional dependencies

### 2. Web-Based Chatbot (`ev_chatbot_web.py`)
**Best for:** Visual exploration and detailed analysis

**Run:**
```bash
streamlit run ev_chatbot_web.py
```

**Features:**
- Beautiful web interface
- Interactive charts and graphs
- Range prediction tool
- Vehicle explorer with filters
- Analytics dashboard

### 3. Demo Script (`demo_chatbot.py`)
**Best for:** Quick overview of capabilities

**Run:**
```bash
python demo_chatbot.py
```

**Features:**
- Pre-programmed demonstration
- Shows all major features
- No user interaction needed

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify data file exists:**
- Ensure `electric_vehicles_spec_2025.csv.csv` is in the same directory

### Running the Chatbot

**Option 1: Command-Line (Recommended for beginners)**
```bash
python ev_chatbot.py
```

**Option 2: Web Interface (Recommended for exploration)**
```bash
streamlit run ev_chatbot_web.py
```

**Option 3: Demo (See it in action)**
```bash
python demo_chatbot.py
```

## 💬 Example Queries

### General Information
- "show stats"
- "overview"
- "how many vehicles?"
- "what is the average range?"

### Search & Discovery
- "show me Tesla"
- "BMW models"
- "top range"
- "best battery"

### Recommendations
- "recommend long range EVs"
- "suggest affordable EVs"
- "recommend"

### Specific Questions
- "what is the average range?"
- "how many brands?"
- "top 5 EVs"

### Help
- "help"
- "what can you do?"

## 🎨 Web Interface Features

### 1. Chat Mode 💬
- Interactive conversation
- Quick action buttons
- Chat history
- Natural language processing

### 2. Range Predictor 🔮
- Input vehicle specifications
- Get instant range predictions
- Compare with similar vehicles
- See how specs affect range

### 3. Analytics Dashboard 📊
- Key metrics overview
- Range distribution charts
- Battery vs Range scatter plots
- Brand comparison tables

### 4. Vehicle Explorer 🔍
- Filter by brand, range, drivetrain
- Sort and search capabilities
- Detailed vehicle information
- Export filtered results

## 📊 Sample Interactions

### Example 1: Finding Top EVs
```
You: top range

Bot: 🏆 Top 5 EVs by Range:
- Mercedes-Benz EQS 450+: 685 km
- Lucid Air Grand Touring: 665 km
- Mercedes-Benz EQS 450 4MATIC: 655 km
- Mercedes-Benz EQS 500 4MATIC: 640 km
- Mercedes-Benz EQS 580 4MATIC: 640 km
```

### Example 2: Brand Search
```
You: show me Tesla

Bot: 🚗 TESLA Electric Vehicles (11 found):
- Model 3 Long Range AWD: 525 km, 75.0 kWh
- Model 3 Long Range RWD: 545 km, 75.0 kWh
- Model S Dual Motor: 575 km, 95.0 kWh
...
```

### Example 3: Statistics
```
You: show stats

Bot: 📊 EV Database Statistics:
- Total vehicles: 478
- Number of brands: 59
- Average range: 393.2 km
- Maximum range: 685 km
- Minimum range: 135 km
- Average battery capacity: 74.0 kWh
```

## 🔮 Range Prediction

The chatbot uses a trained Random Forest model to predict EV range based on:

**Input Features:**
- Battery capacity (kWh)
- Top speed (km/h)
- Efficiency (Wh/km)
- Acceleration (0-100 km/h)
- Towing capacity (kg)
- Torque (Nm)
- Drivetrain (FWD/RWD/AWD)
- Vehicle segment
- Body type

**Model Performance:**
- R² Score: 0.9577 (95.77% accuracy)
- Mean Absolute Error: 15.1 km
- Trained on 478 vehicles

## 🛠️ Technical Details

### Architecture
```
User Input → Natural Language Processing → Query Handler → 
Data Retrieval/ML Prediction → Response Generation → User
```

### Data Source
- 478 electric vehicles from 2025
- 59 different brands
- 22 technical specifications per vehicle
- Source: EV Database (ev-database.org)

### Machine Learning
- Algorithm: Random Forest Regressor
- Training samples: 382 vehicles
- Test samples: 96 vehicles
- Features: 9 key specifications

## 📝 Tips for Best Results

1. **Be specific:** "show me Tesla Model 3" works better than just "Tesla"
2. **Use keywords:** Include words like "top", "best", "recommend", "average"
3. **Ask follow-ups:** The chatbot maintains context
4. **Try variations:** If one query doesn't work, rephrase it
5. **Use the web interface:** For visual exploration and detailed analysis

## 🐛 Troubleshooting

### Common Issues

**Issue:** "File not found" error
**Solution:** Ensure `electric_vehicles_spec_2025.csv.csv` is in the same directory

**Issue:** Import errors
**Solution:** Run `pip install -r requirements.txt`

**Issue:** Streamlit not found
**Solution:** Install with `pip install streamlit plotly`

**Issue:** Chatbot doesn't understand query
**Solution:** Try rephrasing or use "help" to see available commands

## 🎓 Learning Resources

### Understanding the Code
- `ev_chatbot.py` - Command-line interface
- `ev_chatbot_web.py` - Streamlit web interface
- `demo_chatbot.py` - Demonstration script

### Related Files
- `ev_range_predictor.py` - Full ML pipeline
- `quick_ev_analysis.py` - Quick analysis script
- `EV_Range_Prediction.ipynb` - Jupyter notebook

## 🌟 Advanced Usage

### Custom Queries
You can extend the chatbot by adding new query handlers in the `handle_query()` method.

### Integration
The chatbot can be integrated into:
- Web applications
- Mobile apps
- Voice assistants
- Customer service platforms

### API Development
The core functions can be wrapped in a REST API for remote access.

## 📈 Future Enhancements

Potential improvements:
- Voice input/output
- Multi-language support
- Real-time price comparisons
- Charging station finder
- Route planning with range calculation
- User preference learning
- Social features (reviews, ratings)

## 🤝 Support

For questions or issues:
1. Check this guide
2. Run the demo: `python demo_chatbot.py`
3. Try the help command: Type "help" in the chatbot
4. Review the code comments

## 📄 License & Credits

- Dataset: EV Database (ev-database.org)
- ML Model: Scikit-learn Random Forest
- Web Interface: Streamlit
- Visualizations: Plotly

---

**Happy EV Exploring! 🚗⚡**