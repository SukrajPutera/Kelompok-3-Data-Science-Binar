from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import os
from os.path import join
from werkzeug.utils import secure_filename
from flasgger import Swagger, swag_from
import pickle
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

app = Flask(__name__)
CORS(app)

# Add CORS headers to allow all origins
# @app.after_request
# def add_cors_headers(response):
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
#     response.headers['Access-Control-Allow-Methods'] = 'OPTIONS, GET, POST'
#     return response
app.config['CORS_HEADERS']= 'Content-Type'


basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = join(basedir, 'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "API Documentation for Deep Learning",
        "version": "1.0.0",
        "description": "Documentation API for Deep Learning (text)"
    },
    "host": "localhost:5000",  # Update with your host
    "basePath": "/",  # Update if needed
    "schemes": ["http"],
    "consumes": ["application/json"],
    "produces": ["application/json"],
}

swagger = Swagger(app, template=swagger_template)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

sentiment = ['negative', 'neutral', 'positive']

def cleansing(sent):
    strings = sent.lower()
    strings = re.sub(r'[^a-zA-Z0-9]', ' ', strings)
    return strings

with open("lstm/resources/x_pad_sequences.pickle", "rb") as file:
    feature_file_from_lstm = pickle.load(file)

model_file_from_lstm = load_model("lstm/model/model.h5")

with open("rnn/resources/x_pad_sequences.pickle", "rb") as file:
    feature_file_from_rnn = pickle.load(file)

model_file_from_rnn = load_model("rnn/model/model.h5")

@cross_origin()
@swag_from("docs/home.yml", methods=['GET'])
@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Deep Learning API"

@cross_origin()
@swag_from("docs/lstm.yml", methods=['POST'])
@app.route("/lstm", methods=['POST'])
def lstm():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    
    json_response = {
        'status_code': 200,
        'description': 'Results of LSTM model',
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        }
    }
    response_data = jsonify(json_response)
    return response_data

@cross_origin()
@swag_from("docs/lstmCSV.yml", methods=['POST'])
@app.route("/lstmCSV", methods=['POST'])
def lstmCSV():
    original_file = request.files['file']
    filename = secure_filename(original_file.filename)
    filepath = 'static/' + filename
    original_file.save(filepath)
    
    df = pd.read_csv(filepath, header=0)
    
    sentiment_results = []
    for text in df.iloc[:, 0]:
        original_text = text
        text = [cleansing(original_text)]
    
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
        
        prediction = model_file_from_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        sentiment_results.append(get_sentiment)
    
    json_response = {
        'status_code': 200,
        'description': 'Results of LSTM model',
        'data': {
            'text': original_text,
            'sentiment': sentiment_results
        }
    }
    response_data = jsonify(json_response)
    return response_data

@cross_origin()
@swag_from("docs/rnn.yml", methods=['POST'])
@app.route("/rnn", methods=['POST'])
def rnn():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])
    
    prediction = model_file_from_rnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    
    json_response = {
        'status_code': 200,
        'description': 'Results of RNN model',
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        }
    }
    response_data = jsonify(json_response)
    return response_data

@cross_origin()
@swag_from("docs/rnnCSV.yml", methods=['POST'])
@app.route("/rnnCSV", methods=['POST'])
def rnnCSV():
    original_file = request.files['file']
    filename = secure_filename(original_file.filename)
    filepath = 'static/' + filename
    original_file.save(filepath)
    
    df = pd.read_csv(filepath, header=0)
    
    sentiment_results = []
    for text in df.iloc[:, 0]:
        original_text = text
        text = [cleansing(original_text)]
    
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])
        
        prediction = model_file_from_rnn.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        sentiment_results.append(get_sentiment)
    
    json_response = {
        'status_code': 200,
        'description': 'Results of RNN model',
        'data': {
            'text': original_text,
            'sentiment': sentiment_results
        }
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()