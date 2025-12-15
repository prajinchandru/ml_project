import streamlit as st
import pickle
import numpy as np

# --- Configuration ---
# Set the title and icon for the Streamlit app
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰"
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the trained Linear Regression model from the pickle file."""
    try:
        with open('salary_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: 'salary_model.pkl' not found. Please ensure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Streamlit Interface ---

st.title("💰 Simple Salary Predictor")
st.markdown("Use this app to predict salary based on years of experience using a Linear Regression model.")

if model is not None:
    # --- User Input ---
    st.header("1. Input Years of Experience")

    # Slider for user to input years of experience
    years_experience = st.slider(
        'Select the number of years of experience:',
        min_value=0.0,    # Assuming 0 years is the minimum
        max_value=20.0,   # Assuming 20 years is the maximum reasonable experience
        value=5.0,        # Default value
        step=0.5          # Step in 0.5 year increments
    )

    st.info(f"You selected **{years_experience}** years of experience.")

    # --- Prediction Button ---
    if st.button('Predict Salary'):
        try:
            # The model expects a 2D array: [[years_experience]]
            experience_data = np.array([[years_experience]])

            # Make the prediction
            predicted_salary = model.predict(experience_data)[0]

            # --- Output Result ---
            st.header("2. Predicted Salary")
            
            # Format the output for better readability (assuming currency is dollars)
            formatted_salary = f"${predicted_salary:,.2f}"
            
            st.success(f"The predicted salary is approximately:")
            st.balloons() # Confetti/balloon animation for a fun result
            
            st.markdown(f"## **{formatted_salary}**")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

st.sidebar.markdown(
    """
    ### About the Model
    This prediction is based on a simple Linear Regression model trained on 
    a dataset relating 'Years of Experience' to 'Salary'.
    """
)
