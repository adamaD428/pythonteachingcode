from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained Logistic Regression model and scaler using joblib
model = joblib.load('/Users/AdamaDiallo/Desktop/HMI 7540 - Healthcare Info System Development/pythonteachingcode/6 Web Page with Flask/basic page/heart_disease_model.pkl')  # Path to your saved model
scaler = joblib.load('/Users/AdamaDiallo/Desktop/HMI 7540 - Healthcare Info System Development/pythonteachingcode/6 Web Page with Flask/basic page/scaler.pkl')  # Path to your saved scaler

# Route for the home page (where user can input their details)
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions based on user input
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        age = int(request.form['age'])
        sex = 1 if request.form['sex'] == 'Male' else 0  # Encode sex as 1 for Male, 0 for Female
        cholesterol = int(request.form['cholesterol'])
        blood_pressure = int(request.form['blood_pressure'])
        heart_rate = int(request.form['heart_rate'])

        # Prepare the feature array (same format as during training)
        input_data = np.array([[age, sex, cholesterol, blood_pressure, heart_rate]])

        # Scale the input data using the same scaler used during training
        input_scaled = scaler.transform(input_data)

        # Make prediction using the trained model
        prediction = model.predict(input_scaled)

        # Interpret the prediction
        if prediction == 1:
            result = "You have a higher risk of heart disease."
        else:
            result = "You are at a lower risk of heart disease."

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', prediction=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
