import streamlit as st
import joblib
import numpy as np
import os

# Function to load the model and scaler
def load_model():
    # Define the paths for your model and scaler files
    model_path = '/Users/AdamaDiallo/Desktop/HMI 7540 - Healthcare Info System Development/pythonteachingcode/6 Web Page with Flask/basic page/heart_disease_model.pkl'  # Path to your saved model
    scaler_path = '/Users/AdamaDiallo/Desktop/HMI 7540 - Healthcare Info System Development/pythonteachingcode/6 Web Page with Flask/basic page/scaler.pkl'              # Path to your saved scaler

    # Load the model and scaler using joblib
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Streamlit App
def main():
    # Title of the app
    st.title("Heart Disease Prediction")

    # Load the model and scaler
    model, scaler = load_model()

    # Description of the app
    st.write("Enter your details below to check your heart disease risk:")

    # User inputs
    age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cholesterol = st.number_input("Cholesterol Level", min_value=0, max_value=500, value=200, step=1)
    blood_pressure = st.number_input("Blood Pressure (Resting)", min_value=0, max_value=300, value=120, step=1)
    heart_rate = st.number_input("Heart Rate (Max)", min_value=0, max_value=300, value=150, step=1)

    # Encode the 'sex' input (1 for Male, 0 for Female)
    sex_encoded = 1 if sex == "Male" else 0

    # Button for making prediction
    if st.button("Predict"):
        try:
            # Prepare the feature array (same format as during training)
            input_data = np.array([[age, sex_encoded, cholesterol, blood_pressure, heart_rate]])

            # Scale the input data using the same scaler used during training
            input_scaled = scaler.transform(input_data)

            # Make prediction using the trained model
            prediction = model.predict(input_scaled)

            # Interpret the prediction
            if prediction == 1:
                result = "You have a higher risk of heart disease."
            else:
                result = "You are at a lower risk of heart disease."

            # Display the result
            st.success(result)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
