import os
import monpa  # Make sure monpa is installed: pip install monpa
from monpa import utils  # Import the utils for short_sentence
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from pprint import pprint
import pickle


# Helper function to load a stop word list (one stop word per line)
def load_stopwords(filepath):
    stopwords = set()
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords


# Preprocess function using monpa.cut and utils.short_sentence for long texts
def preprocess(text):
    # First, split the long text into short sentences using utils.short_sentence,
    # which handles texts over 200 characters by looking for punctuation ("。", "！", "？", "，")
    short_sentences = utils.short_sentence(text)
    
    # Initialize an empty list for tokens
    tokens = []
    
    # Tokenize each short sentence with monpa.cut and extend the tokens list
    for sentence in short_sentences:
        tokens.extend(monpa.cut(sentence))
    
    # Update path to your stop word list if available
    stopwords_path = 'chinese_stopwords.txt'
    stopwords = load_stopwords(stopwords_path) if os.path.exists(stopwords_path) else set()

    # Optionally: Filter out stop words and tokens that are too short (e.g., length <= 1)
    filtered_tokens = [token for token in tokens if token not in stopwords and len(token) > 1]
    
    return filtered_tokens


# Define the path to your folder containing .txt files (each document is a novel)
folder_path = './literature_traditional'  # Replace with your actual folder path
documents = []

# Read and preprocess each document in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, encoding='utf-8') as file:
            text = file.read()
            tokens = preprocess(text)
            documents.append(tokens)

with open('./stopwords/stopwords_zh.txt', encoding='utf-8') as f:
    remove_words = [line.strip() for line in f if line.strip()]
remove_words.extend(["——", "個子","      我",'      你','      他','      她','        ',])
documents = [[token for token in doc if token not in remove_words] for doc in documents]

# Save the preprocessed documents to a file for later use
with open('documents.pkl', 'wb') as f:
    pickle.dump(documents, f)