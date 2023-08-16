import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Create a stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load slang dictionary
slang_dict = pd.read_csv('new_kamusalay.csv', encoding="ISO-8859-1")
slang_list = list(slang_dict['anakjakartaasikasik'])

# Remove symbols and non-ASCII characters using regex
def remove_symbols(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Keep only letters, numbers, and whitespace
    return text

# Remove stopwords
def remove_stopwords(text, stopwords_list):
    return [w for w in text if w not in stopwords_list]

# Standardize words using the slang dictionary
def standardize(text):
    return [slang_dict['anak jakarta asyik asyik'][slang_list.index(word)] if word in slang_list else word for word in text]

# Stem words
def stemming(text):
    return [stemmer.stem(word) for word in text]

# Main cleaner function
def clean_texts(text, stopwords_list=None):
    '''
    Input: string
    Output: Tokenized and processed words
    '''
    cleaned_text = remove_symbols(text)
    tokenized = cleaned_text.split()  # Split the text into words
    
    if stopwords_list is None:
        removed_stopwords = tokenized
    else:
        removed_stopwords = remove_stopwords(tokenized, stopwords_list)
    
    standardized_words = standardize(removed_stopwords)
    stemmed_words = stemming(standardized_words)
    return stemmed_words

# Testing
testing = "USER ganteng ganteng lg gw sarap \xf0\x9f\x98\x82. Menyampaikan, mengatakan, disuruh"

# Example list of stopwords retrieved from your API (replace with the actual list)
stopwords_list = ["word1", "word2", "word3"]

# Call the clean_texts function
cleaned = clean_texts(testing, stopwords_list)
print(cleaned)
