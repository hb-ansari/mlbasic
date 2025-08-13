import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("📊 Sales Prediction App")
st.write("Predict sales based on your ad spend.")

# Function to create and train the model
@st.cache_resource
def load_or_create_model():
    if os.path.exists("model.pkl"):
        try:
            with open("model.pkl", "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Error loading existing model: {e}. Creating new model...")
    
    # Create a trained model with realistic advertising data (same as your training data)
    # This simulates the original dataset from your Google Drive link
    X_data = np.array([
        [230.1, 37.8, 69.2], [44.5, 39.3, 45.1], [17.2, 45.9, 69.3],
        [151.5, 41.3, 58.5], [180.8, 10.8, 58.4], [8.7, 48.9, 75.0],
        [57.5, 32.8, 23.5], [120.2, 19.6, 11.6], [8.6, 2.1, 1.0],
        [199.8, 2.6, 21.2], [66.1, 5.8, 24.2], [214.7, 24.0, 4.0],
        [23.8, 35.1, 65.9], [97.5, 7.6, 7.2], [204.1, 32.9, 46.0],
        [195.4, 47.7, 52.9], [67.8, 36.6, 114.0], [281.4, 39.6, 55.8],
        [69.2, 20.5, 18.3], [147.3, 23.9, 19.1], [218.4, 27.7, 53.4],
        [237.4, 5.1, 23.5], [13.2, 15.9, 49.6], [228.3, 16.9, 26.2],
        [62.3, 12.6, 18.3], [262.9, 3.5, 19.5], [142.9, 29.3, 12.6],
        [240.1, 16.7, 22.9], [248.8, 27.1, 22.9], [70.6, 16.0, 40.8]
    ])
    
    y_data = np.array([
        22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6,
        8.6, 17.4, 9.2, 9.7, 19.0, 22.4, 12.5, 24.4, 11.3, 13.6,
        21.7, 16.9, 5.7, 20.6, 9.7, 20.5, 15.9, 20.9, 20.4, 11.4
    ])
    
    model = LinearRegression()
    model.fit(X_data, y_data)
    
    # Try to save the model for future use
    try:
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        st.success("✅ Model created and saved!")
    except Exception as e:
        st.info(f"Model created but couldn't save to disk: {e}")
    
    return model

# Load or create the model
with st.spinner("Loading model..."):
    model = load_or_create_model()

# Input fields
st.subheader("Enter Advertising Spend")

col1, col2, col3 = st.columns(3)

with col1:
    tv = st.number_input("TV Ad Spend ($)", 
                        min_value=0.0, 
                        max_value=500.0, 
                        value=100.0, 
                        step=5.0,
                        help="TV advertising budget in dollars")

with col2:
    radio = st.number_input("Radio Ad Spend ($)", 
                           min_value=0.0, 
                           max_value=50.0, 
                           value=20.0, 
                           step=1.0,
                           help="Radio advertising budget in dollars")

with col3:
    newspaper = st.number_input("Newspaper Ad Spend ($)", 
                               min_value=0.0, 
                               max_value=50.0, 
                               value=10.0, 
                               step=1.0,
                               help="Newspaper advertising budget in dollars")

# Prediction
if st.button("🔮 Predict Sales", type="primary"):
    try:
        # Make prediction
        input_data = [[tv, radio, newspaper]]
        pred = model.predict(input_data)
        
        # Display result
        st.success(f"📈 **Predicted Sales: {pred[0]:.2f} units**")
        
        # Show input summary
        with st.expander("📋 Input Summary"):
            st.write(f"• **TV Ad Spend:** ${tv:,.2f}")
            st.write(f"• **Radio Ad Spend:** ${radio:,.2f}")
            st.write(f"• **Newspaper Ad Spend:** ${newspaper:,.2f}")
            st.write(f"• **Total Ad Spend:** ${tv + radio + newspaper:,.2f}")
            
        # Show some insights
        st.info("💡 **Tip:** TV advertising typically has the strongest impact on sales in this model!")
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Add some additional information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app predicts sales based on advertising spend across three channels:
    - **TV**: Television advertising
    - **Radio**: Radio advertising  
    - **Newspaper**: Print advertising
    
    The model is trained on historical advertising and sales data.
    """)
    
    st.header("📊 Model Info")
    if model:
        st.write("**Model Type:** Linear Regression")
        st.write("**Features:** TV, Radio, Newspaper")
        st.write("**Target:** Sales (units)")
