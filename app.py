# # app.py
# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the trained model and scaler
# with open("models/knn_model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# with open("models/scaler.pkl", "rb") as scaler_file:
#     scaler = pickle.load(scaler_file)

# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Retrieve data from form and convert to float for each field
#     data = [
#         float(request.form['age']),
#         float(request.form['gender']),
#         float(request.form['total_bilirubin']),
#         float(request.form['direct_bilirubin']),
#         float(request.form['alkaline_phosphotase']),
#         float(request.form['alamine_aminotransferase']),
#         float(request.form['aspartate_aminotransferase']),
#         float(request.form['total_protiens']),
#         float(request.form['albumin']),
#         float(request.form['albumin_and_globulin_ratio'])
#     ]

#     # Preprocess the input data
#     data_scaled = scaler.transform([data])

#     # Make a prediction
#     prediction = model.predict(data_scaled)
#     result = 'Liver Disease Detected' if prediction[0] == 1 else 'No Liver Disease Detected'

#     return render_template('index.html', prediction_text=result)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import numpy as np
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the trained model and scaler
with open("models/knn_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("models/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from form and convert to float for each field
    data = [
        float(request.form['age']),
        float(request.form['gender']),
        float(request.form['total_bilirubin']),
        float(request.form['direct_bilirubin']),
        float(request.form['alkaline_phosphotase']),
        float(request.form['alamine_aminotransferase']),
        float(request.form['aspartate_aminotransferase']),
        float(request.form['total_protiens']),
        float(request.form['albumin']),
        float(request.form['albumin_and_globulin_ratio'])
    ]

    # Preprocess the input data
    data_scaled = scaler.transform([data])

    # Make a prediction
    prediction = model.predict(data_scaled)
    result = 'Liver Disease Detected' if prediction[0] == 1 else 'No Liver Disease Detected'

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Run on port 5000
