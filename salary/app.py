import streamlit as st
import pickle
import numpy as np

# --- Configuration ---
# Set the title and icon for the Streamlit app
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="centered"
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the trained Linear Regression model from the pickle file."""
    
    # --- IMPORTANT: Set the correct path based on your GitHub repository structure ---
    # The model is inside the 'salary' folder.
    MODEL_PATH = 'salary/salary_model.pkl' 
    
    try:
        # 'rb' means read binary
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        # Display a specific error if the file cannot be found
        st.error(
            f"Error: The model file '{MODEL_PATH}' was not found. "
            "Please ensure it is committed to your GitHub repository in the 'salary' folder."
        )
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model once
model = load_model()

# --- Streamlit Interface ---

st.title("💰 Simple Salary Predictor")
st.markdown("Use this app to predict salary based on years of experience using a Linear Regression model.")

# Only proceed if the model was loaded successfully
if model is not None:
    
    # --- User Input ---
    st.header("1. Input Years of Experience")

    # Slider for user to input years of experience
    years_experience = st.slider(
        'Select the number of years of experience:',
        min_value=0.0,    
        max_value=20.0,   
        value=5.0,        
        step=0.1          # Use 0.1 for finer control
    )

    st.info(f"You selected **{years_experience}** years of experience.")

    # --- Prediction Button ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('🚀 Predict Salary Now', use_container_width=True):
            
            # Prediction logic is wrapped inside the button click
            try:
                # The model expects a 2D array: [[years_experience]]
                experience_data = np.array([[years_experience]])

                # Make the prediction
                predicted_salary = model.predict(experience_data)[0]

                # --- Output Result ---
                st.subheader("2. Predicted Salary Estimate")
                
                # Format the output for better readability (assuming currency is dollars)
                # Ensure the salary is non-negative
                final_salary = max(0, predicted_salary) 
                formatted_salary = f"${final_salary:,.2f}"
                
                st.success(f"Based on the model, the predicted salary is approximately:")
                st.balloons() # Confetti/balloon animation 
                
                # Display the result prominently
                st.markdown(f"<h1 style='text-align: center; color: green;'>{formatted_salary}</h1>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# --- Sidebar Information ---
st.sidebar.header("Model Details")
st.sidebar.info(
    """
    This prediction is based on a simple Linear Regression model. 
    It provides an estimate based on the data it was trained on.
    """
)

# Optional: Display the model's coefficients
if model is not None:
    try:
        # Accessing the model's parameters (assuming LinearRegression)
        intercept = model.intercept_
        coefficient = model.coef_[0]
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Model Parameters")
        st.sidebar.latex(f"Salary = {coefficient:.2f} \cdot (YoE) + {intercept:.2f}")
    except:
        st.sidebar.warning("Could not display model coefficients.")
