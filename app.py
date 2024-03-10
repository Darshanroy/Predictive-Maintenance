from flask import Flask, render_template, request
import pandas as pd
import logging
from Pipelines import PredPipeline

app = Flask(__name__)

# Specify the folder path where the pickle files are stored
folder_path = 'trained_models'

# Load all pickle files from the specified folder
loaded_pickles = PredPipeline.load_pickles_from_folder(folder_path)

# Configure logging to output to the console
logging.basicConfig(level=logging.INFO)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        # Get form data
        form_data = request.form.to_dict()
        logging.info(form_data)

        # Create a DataFrame from the form data
        processed_data = pd.DataFrame([form_data], columns=['Type', 'Air temperature (K)', 'Process temperature (K)',
                                               'Rotational speed (rpm)', 'Torque (Nm)', 'Tool wear (min)',
                                               'Heat Dissipation Failure', 'No Failure', 'Overstrain Failure',
                                               'Power Failure', 'Random Failures', 'Tool Wear Failure'])

        logging.info(processed_data)

        # Make predictions using the loaded models
        predictions = {}
        for model_name, model in loaded_pickles.items():
            try:
                prediction = model.predict(processed_data)
                predictions[model_name] = prediction
                logging.info(f"Predictions using {model_name}: {prediction}")
            except Exception as e:
                logging.error(f"Error predicting with {model_name}: {str(e)}")

        # Render the template with predictions
        return render_template('prediction_results.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
