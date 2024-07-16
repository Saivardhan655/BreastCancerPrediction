import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

reg_model = pickle.load(open('breast_cancer.pkl', 'rb'))

q = {1: 'Malignant', 0: 'Benign'}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def prediction_api():
    data = request.json['data']
    print(data)

    new_data = np.array(list(data.values())).reshape(1, -1)
    print(new_data)
    
    output = reg_model.predict(new_data)
    
    prediction = q[output[0]]
    return jsonify({'prediction': prediction})
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    data=np.array(data)
    output = reg_model.predict(data.reshape(1,-1))
    prediction = q[output[0]]
    return render_template("home.html",prediction_text="The predicted is {}".format(prediction))



if __name__ == "__main__":
    app.run(debug=True)
