from flask import Flask, jsonify, request
from flasgger import Swagger, swag_from, LazyString, LazyJSONEncoder
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import pickle
from keras.models import load_model
import os
from werkzeug.utils import secure_filename
from os.path import join
from cleaner import clean_texts  # Import the clean_texts function

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info={
        'title': LazyString(lambda: 'API Documentation for Sentiment Analysis API'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda: 'Dokumentasi API untuk Sentiment Analysis API'),
    },
    host=LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)

# Custom homepage route
@app.route('/', methods=['GET'])
def home():
    welcome_msg = {
        "version": "1.0.0",
        "message": "Welcome to Flask Sentiment Analysis",
        "author": "Kelompok 3"
    }
    return jsonify(welcome_msg)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

# Set num_words for the tokenizer
num_words = 10000  # Use the same value as in your training code
tokenizer = Tokenizer(num_words=num_words, split=' ', lower=True)

# Load models and feature files
model_file_from_lstm = load_model("lstm/model/model.h5")
with open("lstm/resources/x_pad_sequences.pickle", "rb") as file:
    feature_file_from_lstm = pickle.load(file)

model_file_from_rnn = load_model("rnn/model/model.h5")
with open("rnn/resources/x_pad_sequences.pickle", "rb") as file:
    feature_file_from_rnn = pickle.load(file)

sentiment = ['negative', 'neutral', 'positive']

# Load models and feature files
model_file_from_lstm = load_model("lstm/model/model.h5")
with open("lstm/resources/x_pad_sequences.pickle", "rb") as file:
    feature_file_from_lstm = pickle.load(file)

model_file_from_rnn = load_model("rnn/model/model.h5")
with open("rnn/resources/x_pad_sequences.pickle", "rb") as file:
    feature_file_from_rnn = pickle.load(file)

# @cross_origin()
@swag_from("docs/lstm.yml", methods=['POST'])
@app.route("/lstm", methods=['POST'])
def lstm():
    try:
        original_text = request.form.get('text')        
        if original_text is not None:
            cleaned_text = ' '.join(clean_texts(original_text))  # Clean the text
            text = [cleaned_text]
            
            # Tokenize and pad the text to have a length of 96 tokens
            feature = tokenizer.texts_to_sequences(text)
            feature = pad_sequences(feature, maxlen=96)  # Use the same maxlen as in training
            
            prediction = model_file_from_lstm.predict(feature)
            get_sentiment = sentiment[np.argmax(prediction[0])]

            json_response = {
                'status_code': 200,
                'description': 'Results of LSTM model',
                'data': {
                    'text': original_text,
                    'cleaned_text': cleaned_text,
                    'sentiment': get_sentiment
                }
            }
        else:
            json_response = {
                'status_code': 400,
                'error': 'Missing or invalid "text" in request data'
            }
        
        response_data = jsonify(json_response)
        return response_data
    
    except Exception as e:
        json_response = {
            'status_code': 500,
            'error': str(e)
        }
        response_data = jsonify(json_response)
        return response_data

# @cross_origin()
@swag_from("docs/rnn.yml", methods=['POST'])
@app.route("/rnn", methods=['POST'])
def rnn():
    original_text = request.json.get('text')  # Use request.json instead of request.form
    if original_text is None:
        json_response = {
            'status_code': 400,
            'error': 'Missing or invalid "text" in request data'
        }
        response_data = jsonify(json_response)
        return response_data
    
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

# @cross_origin()
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