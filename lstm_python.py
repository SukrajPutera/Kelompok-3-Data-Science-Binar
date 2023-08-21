import re
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd


#Load the LSTM model
cat_sent = ['negative','neutral', 'positive']
model = load_model('LSTM/model.h5')

max_features = 10000

tokenizer_file = open('LSTM/tokenizer.pickle', 'rb')
tokenizer = pickle.load(tokenizer_file)

file = open('LSTM/x_pad_sequences.pickle' , 'rb')
X = pickle.load(file)


#Cleansing Function
def text_cleansing(text):
    clean_text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))|([#@]\S+)|user|\n|\t', ' ', text)
    clean_text = re.sub(r'[^\w\s]', '',clean_text)
    clean_text = clean_text.lower()
    return clean_text

#Create Function for LSTM text input
def model_lstm(text):
    #Text Cleansing
    clean_text = [text_cleansing(text)]

    predicted = tokenizer.texts_to_sequences(clean_text)
    guess = pad_sequences(predicted, maxlen=X.shape[1])
    prediction = model.predict(guess)
    polarity = np.argmax(prediction[0])
    sentiment = cat_sent[polarity]

    return sentiment

#Create Function for LSTM file upload 
def lstm_upload(file_upload):
    
    # Read csv file upload using encoding latin-1
    lstm_file = pd.read_csv(file_upload)

    lstm_file = pd.DataFrame(lstm_file.iloc[:,0])

    # Rename column to "text" 
    lstm_file.columns = ["text"]
    
    # Apply the LSTM sentiment prediction to the text
    lstm_file['clean_text'] = lstm_file.apply(lambda row : text_cleansing(row['text']), axis = 1)
    lstm_file['sentiment'] = lstm_file.apply(lambda row : model_lstm(row['text']), axis = 1)

    return lstm_file