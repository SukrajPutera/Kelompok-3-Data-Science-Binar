import re
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

from lstm_python import text_cleansing, model_lstm, lstm_upload
from nn_python import neural_network_model, neural_network_upload, text_cleansing

app = Flask(__name__)

app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info= {
        'title': LazyString(lambda: 'API Documentation for Sentiment Analysis API'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda: 'Dokumentasi API untuk Sentiment Analysis API'),
        'Author': LazyString(lambda: 'Kelompok 3 : Amanda Risfa dan Sukraj Putera'),
    },
    host= LazyString(lambda: request.host) 
)

swagger_config = {
    'headers': [],
    'specs': [
        {
            'endpoint': 'docs',
            'route': '/docs.json',
        }
    ],
    'static_url_path': '/flasgger_static',
    'swagger_ui': True,
    'specs_route': '/docs/'

}

swagger = Swagger(app, template=swagger_template, 
                  config=swagger_config)


# Custom homepage route
@swag_from('docs/home.yml', methods=['GET'])
@app.route('/', methods=['GET'])
def home():
    welcome_msg = {
        "version": "1.0.0",
        "message": "Welcome to Flask Sentiment Analysis",
        "author": "Amanda Risfa & Sukraj Putera"
    }
    return jsonify(welcome_msg)


#LSTM 
@swag_from('docs/lstm.yml', methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm_form():

   
    raw_text = request.form["raw_text"]
    clean_text = text_cleansing(raw_text)

 
    results = model_lstm(clean_text)
    result_response = {"text_clean": clean_text, "results": results}
    return jsonify(result_response)

@swag_from('docs/lstmCSV.yml', methods=['POST'])
@app.route('/lstmCSV', methods=['POST'])
def LSTM_upload():

    # Get file from upload to dataframe
    uploaded_file = request.files['upload_file']

    # Read csv file to dataframe the analyize the sentiment
    df_lstm = lstm_upload(uploaded_file)
    result_response = df_lstm.T.to_dict()
    
    return jsonify(result_response)


#Neural Network 
@swag_from('docs/nn.yml', methods=['POST'])
@app.route('/nn', methods=['POST'])
def neural_network_form():

    raw_text = request.form["raw_text"]
    clean_text = text_cleansing(raw_text)

  
    results = neural_network_model(clean_text)
    result_response = {"text_clean": clean_text, "results": results}

    return jsonify(result_response)


@swag_from('docs/nnCSV.yml', methods=['POST'])
@app.route('/nnCSV', methods=['POST'])
def Neural_Network_upload():

    # Get file from upload to dataframe
    uploaded_file = request.files['upload_file']

    # Read csv file to dataframe the analyize the sentiment
    df_nn = neural_network_upload(uploaded_file)
    result_response = df_nn.T.to_dict()

    return jsonify(result_response)


if __name__ == '__main__':
    app.run()