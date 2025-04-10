# Importing necessary modules
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler

# Flask application setup
application = Flask(__name__)
app = application

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load models and scaler
try:
    ridge_model = pickle.load(open('notebooks/ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('notebooks/scaler.pkl', 'rb'))
    logging.info("Model and scaler loaded successfully.")
except FileNotFoundError:
    logging.error("Required model files not found in the 'models' directory.")
    raise

# Define routes
@app.route('/')
def index():
    """Render the index page."""
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Handle prediction requests."""
    if request.method == 'POST':
        try:
            # Parse and validate input data
            Temperature = request.form.get('Temperature')
            RH = request.form.get('RH')
            Ws = request.form.get('Ws')
            Rain = request.form.get('Rain')
            FFMC = request.form.get('FFMC')
            DMC = request.form.get('DMC')
            ISI = request.form.get('ISI')
            Classes = request.form.get('Classes')
            Region = request.form.get('Region')

            # Check for missing inputs
            if any(val is None or val.strip() == '' for val in [Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]):
                return render_template('home.html', error="All fields are required.")
            
            # Convert inputs to floats
            Temperature = float(Temperature)
            RH = float(RH)
            Ws = float(Ws)
            Rain = float(Rain)
            FFMC = float(FFMC)
            DMC = float(DMC)
            ISI = float(ISI)
            Classes = float(Classes)
            Region = float(Region)

            # Prepare input data
            input_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]

            # Address StandardScaler warning (if applicable)
            feature_names = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
            input_df = pd.DataFrame(input_data, columns=feature_names)

            # Scale the input data
            new_data_scaled = standard_scaler.transform(input_df)

            # Predict using the trained model
            result = ridge_model.predict(new_data_scaled)

            # Log prediction result
            logging.info(f"Prediction result: {result[0]}")

            # Interpret FWI value based on dataset-specific information
            # Non-fire cases average: 0.96, Fire cases average: 11.73
            fwi_value = result[0]
            
            # Define dataset-specific thresholds
            low_threshold = 2.0        # Very low risk (well below fire average)
            moderate_threshold = 5.0   # Moderate risk (between non-fire and fire averages)
            high_threshold = 8.0       # High risk (closer to fire average)
            very_high_threshold = 11.0 # Very high risk (near fire average)
            # Anything above very_high_threshold is extreme risk
            
            if fwi_value < low_threshold:
                risk_level = "safe"
                fire_message = None  # No warning for low risk
            elif fwi_value < moderate_threshold:
                risk_level = "Moderate"
                fire_message = None  # No warning for moderate risk
            elif fwi_value < high_threshold:
                risk_level = "High"
                fire_message = "WARNING: Elevated risk of forest fire detected!"
            elif fwi_value < very_high_threshold:
                risk_level = "Very High"
                fire_message = "ALERT: High risk of forest fire detected! Take precautions!"
            else:
                risk_level = "Extreme"
                fire_message = "DANGER: Fire conditions detected! Immediate attention required!"

            # Return the result with appropriate warnings
            return render_template('home.html', 
                                  results=fwi_value, 
                                  risk_level=risk_level,
                                  fire_warning=fire_message)

        except ValueError as ve:
            # Handle invalid numeric input
            logging.error(f"ValueError: {ve}")
            return render_template('home.html', error="Invalid input. Please ensure all fields contain numeric values.")
        
        except Exception as e:
            # Handle any other errors
            logging.error(f"Error: {e}")
            return render_template('home.html', error=f"An unexpected error occurred: {str(e)}")

    # Handle GET requests or initial page load
    return render_template('home.html')

# Main driver function
if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)