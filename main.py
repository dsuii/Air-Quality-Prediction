import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data():
    return pd.read_csv("air quality prediction.csv")

data = load_data()

if data.empty:
    st.error("The dataset is empty. Please check the file.")
    st.stop()

# Preprocessing the data
features = ["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)"]
target = "PM2.5 AQI Value"

X = data[features].copy()  
y = data[target]

# Handle missing values by filling with median
X.fillna(X.median(), inplace=True)
y.fillna(y.median(), inplace=True)

# Predictive Modeling
st.title("Air Quality Prediction")

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training with XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Predictions with XGBoost
xgb_y_pred = xgb_model.predict(X_test_scaled)

# Model Evaluation for XGBoost
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_rmse = np.sqrt(xgb_mse)
xgb_r2 = r2_score(y_test, xgb_y_pred)

# Display Evaluation Metrics
st.write("### Model Evaluation")

# XGBoost
st.write("#### XGBoost")
st.write(f"Mean Squared Error (MSE): {xgb_mse}")
st.write(f"Root Mean Squared Error (RMSE): {xgb_rmse}")
st.write(f"R2 Score: {xgb_r2}")

# Plot predictions vs actual for XGBoost
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, xgb_y_pred, alpha=0.7, color="purple")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black", lw=2)
ax.set_title("XGBoost - Predicted vs Actual PM2.5 AQI")
ax.set_xlabel("Actual PM2.5 AQI")
ax.set_ylabel("Predicted PM2.5 AQI")
st.pyplot(fig)

# Dashboard title and description
st.title("Enhanced Air Quality Dashboard")
st.markdown("""
This dashboard provides a detailed analysis of air quality metrics, including:
- **Predictive Modeling**: Predict PM2.5 AQI values using the XGBoost model.
- **Pollutant Breakdown**: Visualize AQI values for major pollutants like CO, Ozone, NO2, and PM2.5.
- **Historical Trends**: Track how AQI values change over time.
- **Environmental Factors**: Analyze temperature, humidity, and wind speed relationships with AQI.
- **Simulated Future Forecasts**: Forecast future PM2.5 values based on historical trends.
""")

# Pollutant Breakdown
st.header("Pollutant Breakdown")
pollutant_columns = [
    "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"
]
if st.button("Show Pollutant Breakdown"):
    avg_values = data[pollutant_columns].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_values.plot(kind="bar", color=["#FFA07A", "#20B2AA", "#87CEEB", "#FF6347"], ax=ax)
    ax.set_title("Average AQI Values for Pollutants")
    ax.set_ylabel("AQI Value")
    ax.set_xlabel("Pollutants")
    st.pyplot(fig)

# Historical Trends
st.header("Historical Trends")
selected_pollutant = st.selectbox("Select a pollutant to view trends", pollutant_columns)
if selected_pollutant:
    st.markdown(f"### Historical Trend: {selected_pollutant}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data.index, data[selected_pollutant], label=selected_pollutant, color="blue")
    ax.set_title(f"Historical Trend for {selected_pollutant}")
    ax.set_xlabel("Time (Index)")
    ax.set_ylabel("AQI Value")
    ax.legend()
    st.pyplot(fig)

# Environmental Factors Analysis
st.header("Environmental Factors")
factor = st.selectbox("Select a factor to analyze against AQI", ["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)"])
if factor:
    st.markdown(f"### {factor} vs AQI Value")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data[factor], data["AQI Value"], alpha=0.7, color="green")
    ax.set_title(f"{factor} vs AQI Value")
    ax.set_xlabel(factor)
    ax.set_ylabel("AQI Value")
    st.pyplot(fig)

# Future Forecasts
st.header("Simulated Future Forecasts")
if st.button("Generate Future Forecast"):
    st.markdown("### Simulated PM2.5 AQI Forecast")
    future_dates = pd.date_range(start="2024-01-01", periods=12, freq="ME")  # Use 'ME' for Month End
    simulated_values = np.random.normal(loc=data["PM2.5 AQI Value"].mean(), scale=10, size=len(future_dates))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(future_dates, simulated_values, label="Simulated PM2.5 AQI", color="purple")
    ax.set_title("Future PM2.5 AQI Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("AQI Value")
    ax.legend()
    st.pyplot(fig)
