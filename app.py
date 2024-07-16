import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model
reg_model = pickle.load(open('breast_cancer.pkl', 'rb'))

# Mapping of the numerical output to the corresponding diagnosis
q = {1: 'Malignant', 0: 'Benign'}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def prediction_api():
    data = request.json['data']
    print(data)
    
    # Convert the input data into the correct shape for prediction
    new_data = np.array(list(data.values())).reshape(1, -1)  # Reshape to (1, number_of_features)
    print(new_data)
    
    # Predict using the loaded model
    output = reg_model.predict(new_data)
    
    # Map the output to the corresponding diagnosis and return as JSON
    prediction = q[output[0]]
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
