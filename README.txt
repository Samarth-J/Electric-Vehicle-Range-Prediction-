🧩 Problem Statement

The goal is to develop a machine learning model that can predict the real-world driving range (in km) of an electric vehicle based on its technical and physical specifications.

📘 Background

Electric vehicle (EV) range is one of the most critical factors influencing consumer choice and market competitiveness. The range depends on multiple factors such as battery capacity, vehicle weight, efficiency, drivetrain type, and aerodynamic design. By leveraging the specifications of various EVs, we can build a predictive model to estimate range — helping manufacturers benchmark designs and customers compare vehicles more effectively.

🎯 Objective

Build a regression model that predicts the range_km of an EV given the following features:

Numerical features:

battery_capacity_kWh

top_speed_kmh

efficiency_wh_per_km

acceleration_0_100_s

towing_capacity_kg

length_mm, width_mm, height_mm

torque_nm

Categorical features:

brand, battery_type, fast_charge_port, drivetrain, segment, car_body_type

📊 Expected Output

A trained regression model (e.g., Random Forest, XGBoost, or Neural Network) that predicts range_km.

Model evaluation using metrics such as MAE, RMSE, and R².

Feature importance analysis to interpret which specifications most affect the range.

🚀 Possible Extensions

Predict efficiency (Wh/km) instead of range.

Cluster EVs into market segments based on performance and size.

Build a recommendation system suggesting EVs similar to a given model.