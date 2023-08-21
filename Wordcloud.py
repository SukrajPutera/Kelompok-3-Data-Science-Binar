import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load and preprocess the data
df = pd.read_csv('train_preprocess.tsv.txt', sep='\t')
df.columns = ['Text', 'Sentiment']

# Download NLTK stopwords data
import nltk
nltk.download('punkt')

# Combine all the cleaned text into a single string
combined_text = ' '.join(df['Text'])

# Tokenize the combined text
words = word_tokenize(combined_text)

# Create a custom stopwords list for Indonesian language
custom_stop_words = set(["dan", "di", "sini", "..."])  # Add more stopwords as needed

# Remove custom stopwords from the list of words
filtered_words = [word for word in words if word.lower() not in custom_stop_words]

# Join the filtered words back into a single string
filtered_text = ' '.join(filtered_words)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
