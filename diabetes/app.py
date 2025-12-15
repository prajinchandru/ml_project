import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Configuration ---
MODEL_PATH = 'linear_reg_model.pkl'

# --- Load the Model ---
# The model uses the features: age, mass, insu, plas 
# The classes are: tested_negative, tested_positive 
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    st.sidebar.success("Logistic Regression Model loaded successfully! (scikit-learn 1.6.1)")
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it is in the same directory.")
    model = None
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

# --- Streamlit Application ---
st.title("🩺 Diabetes Prediction App")
st.markdown("This application uses a Logistic Regression model to predict diabetes based on patient data.")
st.markdown("---")

if model:
    # --- Input Form ---
    with st.form("prediction_form"):
        st.header("Patient Input Data")

        # The model features are 'age', 'mass', 'insu', 'plas' 
        
        # 1. Plasma Glucose Concentration (plas)
        plas = st.slider("Plasma Glucose Concentration (mg/dL)", 0, 200, 100, help="Two-hour plasma glucose concentration.")
        
        # 2. Body Mass Index (mass)
        mass = st.slider("Body Mass Index (BMI)", 15.0, 50.0, 25.0, step=0.1, help="Weight (kg) / [Height (m)]^2.")
        
        # 3. Two-Hour Serum Insulin (insu)
        insu = st.slider("Two-Hour Serum Insulin (mu U/ml)", 0, 850, 80, help="Serum insulin concentration after a two-hour oral glucose tolerance test.")
        
        # 4. Age (age)
        age = st.slider("Age (Years)", 20, 80, 30)

        # Submit Button
        submitted = st.form_submit_button("Predict Outcome")

    # --- Prediction Logic ---
    if submitted:
        # Create a DataFrame from the user inputs, matching the model's expected features
        input_data = pd.DataFrame({
            'age': [age],
            'mass': [mass],
            'insu': [insu],
            'plas': [plas]
        })

        # Make the prediction
        try:
            # The model was trained with classes 'tested_negative' and 'tested_positive' 
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # Format output
            st.markdown("---")
            st.header("Prediction Result")
            
            if prediction == 'tested_positive':
                # Get the probability for the positive class
                prob_positive = probability[model.classes_ == 'tested_positive'][0]
                st.error(f"**Prediction: Tested Positive for Diabetes** 😟")
                st.metric(label="Confidence (Positive)", value=f"{prob_positive * 100:.2f}%")
            else:
                # Get the probability for the negative class
                prob_negative = probability[model.classes_ == 'tested_negative'][0]
                st.success(f"**Prediction: Tested Negative for Diabetes** 😄")
                st.metric(label="Confidence (Negative)", value=f"{prob_negative * 100:.2f}%")

            st.markdown("---")
            st.subheader("Input Data Used for Prediction")
            st.dataframe(input_data)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.info("The application cannot run without a successfully loaded model. Please check the `linear_reg_model.pkl` file and ensure `scikit-learn==1.6.1` is installed.")
