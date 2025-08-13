import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("📊 Sales Prediction App")
st.write("Predict sales based on your ad spend.")

# Create model if it doesn't exist
if not os.path.exists("model.pkl"):
    with st.spinner("Creating model for first run..."):
        # Create a trained model with realistic advertising data
        X_data = np.array([
            [230.1, 37.8, 69.2], [44.5, 39.3, 45.1], [17.2, 45.9, 69.3],
            [151.5, 41.3, 58.5], [180.8, 10.8, 58.4], [8.7, 48.9, 75.0],
            [57.5, 32.8, 23.5], [120.2, 19.6, 11.6], [8.6, 2.1, 1.0],
            [199.8, 2.6, 21.2], [66.1, 5.8, 24.2], [214.7, 24.0, 4.0]
        ])
        y_data = np.array([22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6, 8.6, 17.4])
        
        model = LinearRegression()
        model.fit(X_data, y_data)
        
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Input fields
tv = st.number_input("TV Ad Spend", 0.0, 500.0, 100.0)
radio = st.number_input("Radio Ad Spend", 0.0, 50.0, 20.0)
newspaper = st.number_input("Newspaper Ad Spend", 0.0, 50.0, 10.0)

if st.button("Predict Sales"):
    pred = model.predict([[tv, radio, newspaper]])
    st.success(f"Predicted Sales: {pred[0]:.2f} units")
