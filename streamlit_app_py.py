import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime

st.title("📊 Advanced Sales Prediction App")
st.write("Predict sales based on your ad spend with confidence intervals and analytics.")

# Function to create and train the model with scaling
@st.cache_resource
def load_or_create_model():
    model_file = "sales_model.joblib"
    scaler_file = "scaler.joblib"
    
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        try:
            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            return model, scaler, None, None
        except Exception as e:
            st.warning(f"Error loading existing model: {e}. Creating new model...")
    
    # Enhanced training data (same as before but with more samples)
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
    
    # Apply scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y_data)
    
    # Calculate residuals for confidence intervals
    y_pred_train = model.predict(X_scaled)
    residuals = y_data - y_pred_train
    std_error = np.std(residuals)
    
    # Try to save the model and scaler
    try:
        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)
        st.success("✅ Model and scaler created and saved!")
    except Exception as e:
        st.info(f"Model created but couldn't save to disk: {e}")
    
    return model, scaler, X_data, y_data

# Function to save prediction history
def save_prediction_history(tv, radio, newspaper, prediction, confidence_low, confidence_high):
    history_file = "prediction_history.csv"
    
    new_entry = pd.DataFrame({
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "TV": [tv],
        "Radio": [radio], 
        "Newspaper": [newspaper],
        "Total_Spend": [tv + radio + newspaper],
        "Predicted_Sales": [prediction],
        "Confidence_Low": [confidence_low],
        "Confidence_High": [confidence_high]
    })
    
    try:
        if os.path.exists(history_file):
            existing_data = pd.read_csv(history_file)
            updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
        else:
            updated_data = new_entry
            
        updated_data.to_csv(history_file, index=False)
        return True
    except Exception as e:
        st.error(f"Could not save prediction history: {e}")
        return False

# Load or create the model
with st.spinner("Loading model..."):
    model, scaler, X_data, y_data = load_or_create_model()

# Store in session state for faster access
if 'model' not in st.session_state:
    st.session_state.model = model
    st.session_state.scaler = scaler

# Calculate standard error for confidence intervals
if X_data is not None and y_data is not None:
    X_scaled = st.session_state.scaler.transform(X_data)
    y_pred_train = st.session_state.model.predict(X_scaled)
    residuals = y_data - y_pred_train
    std_error = np.std(residuals)
else:
    std_error = 2.5  # Default fallback

# Input section
st.subheader("🎯 Enter Advertising Spend")

# Quick example buttons
st.write("**Quick Examples:**")
col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)

if col_ex1.button("📺 High TV Focus", help="TV-focused campaign"):
    st.session_state.tv_spend = 200.0
    st.session_state.radio_spend = 15.0
    st.session_state.newspaper_spend = 10.0

if col_ex2.button("📻 Balanced Mix", help="Balanced advertising"):
    st.session_state.tv_spend = 120.0
    st.session_state.radio_spend = 25.0
    st.session_state.newspaper_spend = 20.0

if col_ex3.button("📰 Print Heavy", help="Newspaper-focused"):
    st.session_state.tv_spend = 80.0
    st.session_state.radio_spend = 20.0
    st.session_state.newspaper_spend = 40.0

if col_ex4.button("🚀 Maximum Spend", help="High budget campaign"):
    st.session_state.tv_spend = 300.0
    st.session_state.radio_spend = 45.0
    st.session_state.newspaper_spend = 50.0

# Input fields with session state
col1, col2, col3 = st.columns(3)

with col1:
    tv = st.number_input("TV Ad Spend ($)", 
                        min_value=0.0, 
                        max_value=500.0, 
                        value=st.session_state.get('tv_spend', 100.0), 
                        step=5.0,
                        help="TV advertising budget in dollars")

with col2:
    radio = st.number_input("Radio Ad Spend ($)", 
                           min_value=0.0, 
                           max_value=50.0, 
                           value=st.session_state.get('radio_spend', 20.0), 
                           step=1.0,
                           help="Radio advertising budget in dollars")

with col3:
    newspaper = st.number_input("Newspaper Ad Spend ($)", 
                               min_value=0.0, 
                               max_value=50.0, 
                               value=st.session_state.get('newspaper_spend', 10.0), 
                               step=1.0,
                               help="Newspaper advertising budget in dollars")

# Real-time preview (without saving)
total_spend = tv + radio + newspaper
if total_spend > 0:
    # Show spending distribution pie chart
    st.subheader("💰 Ad Spend Distribution")
    
    if tv > 0 or radio > 0 or newspaper > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = [tv, radio, newspaper]
        labels = ["TV", "Radio", "Newspaper"]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Filter out zero values
        non_zero = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
        if non_zero:
            sizes_filtered, labels_filtered, colors_filtered = zip(*non_zero)
            
            wedges, texts, autotexts = ax.pie(sizes_filtered, 
                                            labels=labels_filtered, 
                                            autopct='%1.1f%%',
                                            colors=colors_filtered,
                                            startangle=90,
                                            explode=[0.05] * len(sizes_filtered))
            
            # Enhance text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
            
            ax.set_title(f'Total Budget: ${total_spend:,.2f}', fontsize=14, weight='bold')
            st.pyplot(fig)
            plt.close()

# Prediction section
if st.button("🔮 Predict Sales", type="primary", use_container_width=True):
    if total_spend == 0:
        st.error("Please enter at least some advertising spend!")
    else:
        try:
            # Scale input data
            input_data = np.array([[tv, radio, newspaper]])
            input_scaled = st.session_state.scaler.transform(input_data)
            
            # Make prediction
            pred = st.session_state.model.predict(input_scaled)
            prediction = pred[0]
            
            # Calculate confidence interval
            confidence_low = prediction - 1.96 * std_error
            confidence_high = prediction + 1.96 * std_error
            
            # Display main result with metric
            st.subheader("📈 Prediction Results")
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric(
                    label="Predicted Sales", 
                    value=f"{prediction:.2f} units",
                    help="Point estimate from the linear regression model"
                )
            
            with col_metric2:
                st.metric(
                    label="Confidence Range", 
                    value=f"{confidence_high - confidence_low:.2f}",
                    delta=f"±{1.96 * std_error:.2f}",
                    help="95% confidence interval width"
                )
            
            with col_metric3:
                avg_sales = 14.0  # Approximate average from training data
                delta_vs_avg = prediction - avg_sales
                st.metric(
                    label="vs Average", 
                    value=f"{avg_sales:.1f} units",
                    delta=f"{delta_vs_avg:+.1f}",
                    help="Comparison to average sales"
                )
            
            # Confidence interval display
            st.info(f"📊 **95% Confidence Interval:** {confidence_low:.2f} - {confidence_high:.2f} units")
            
            # Visual comparison
            st.subheader("📊 Sales Comparison")
            comparison_data = {
                'Category': ['Predicted Sales', 'Average Sales', 'Confidence Low', 'Confidence High'],
                'Value': [prediction, avg_sales, confidence_low, confidence_high],
                'Color': ['#FF6B6B', '#95A5A6', '#F39C12', '#F39C12']
            }
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(comparison_data['Category'], comparison_data['Value'], 
                         color=comparison_data['Color'], alpha=0.8)
            ax.set_ylabel('Sales (units)')
            ax.set_title('Sales Prediction Analysis')
            
            # Add value labels on bars
            for bar, value in zip(bars, comparison_data['Value']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Input summary
            with st.expander("📋 Input Summary"):
                summary_data = {
                    'Channel': ['TV', 'Radio', 'Newspaper', 'TOTAL'],
                    'Spend ($)': [f'{tv:,.2f}', f'{radio:,.2f}', f'{newspaper:,.2f}', f'{total_spend:,.2f}'],
                    'Percentage': [f'{(tv/total_spend)*100:.1f}%' if total_spend > 0 else '0%',
                                 f'{(radio/total_spend)*100:.1f}%' if total_spend > 0 else '0%',
                                 f'{(newspaper/total_spend)*100:.1f}%' if total_spend > 0 else '0%',
                                 '100.0%']
                }
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            # ROI Analysis
            if total_spend > 0:
                roi = (prediction / total_spend) * 100
                st.success(f"📈 **Estimated ROI:** {roi:.1f}% (Sales per dollar spent)")
            
            # Save prediction history
            save_prediction_history(tv, radio, newspaper, prediction, confidence_low, confidence_high)
            
            # Insights
            st.subheader("💡 Insights")
            insights = []
            
            if tv > radio + newspaper:
                insights.append("🔥 **TV-Heavy Strategy:** Your campaign is TV-focused, which typically has strong reach.")
            if radio > tv and radio > newspaper:
                insights.append("📻 **Radio-Focused:** Radio can be cost-effective for local targeting.")
            if newspaper > 30:
                insights.append("📰 **High Print Spend:** Consider digital alternatives for better targeting.")
            if total_spend > 200:
                insights.append("💰 **High Budget Campaign:** Make sure to track performance closely.")
            if prediction > avg_sales + std_error:
                insights.append("🚀 **Above Average Performance:** This combination looks promising!")
                
            for insight in insights:
                st.write(insight)
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Sidebar with additional info and history
with st.sidebar:
    st.header("ℹ️ About This App")
    st.write("""
    This advanced sales prediction app uses:
    - **Scaled Linear Regression** for better accuracy
    - **Confidence intervals** for prediction reliability
    - **Visual analytics** for better insights
    - **Prediction history** tracking
    """)
    
    st.header("📊 Model Details")
    st.write("**Model Type:** Linear Regression with StandardScaler")
    st.write("**Features:** TV, Radio, Newspaper (scaled)")
    st.write("**Target:** Sales (units)")
    st.write(f"**Standard Error:** ±{std_error:.2f} units")
    
    # Show recent predictions if available
    if os.path.exists("prediction_history.csv"):
        st.header("📈 Recent Predictions")
        try:
            history = pd.read_csv("prediction_history.csv")
            if len(history) > 0:
                recent = history.tail(5)[['TV', 'Radio', 'Newspaper', 'Predicted_Sales']]
                st.dataframe(recent, use_container_width=True)
                
                if st.button("📥 Download Full History"):
                    csv = history.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="sales_predictions_history.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.write("History unavailable")
    
    st.header("🎯 Tips")
    st.write("""
    - **TV** typically has the strongest impact
    - **Radio** offers good cost efficiency
    - **Newspaper** has diminishing returns
    - Try different combinations with the example buttons!
    """)
