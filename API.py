from flask import Flask, request, jsonify
import pickle
import sklearn
import numpy as np
model = pickle.load(open('C:\\Users\\koustubh kulkarni\\LOVE\\Predictor\\SVM_MODEL_IRIS', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    sl = request.form.get('sl')
    sw = request.form.get('sw')
    pl = request.form.get('pl')
    pw = request.form.get('pw')
    input_query = np.array([sl, sw, pl, pw]).reshape(1, -1)
    Outii = model.predict(input_query)[0]
    return jsonify({'Species': str(Outii)})


if __name__ == '__main__':
    app.run(debug=True)
