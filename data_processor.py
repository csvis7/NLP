import os

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    return " ".join(filtered_tokens)

def process_data(directory="data"):
    all_data = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            topic = filename.replace(".txt", "").capitalize()
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    cleaned = clean_text(line.strip())
                    if cleaned:
                        all_data.append(cleaned)
                        labels.append(topic)
    
    return all_data, labels

if __name__ == "__main__":
    data, labels = process_data()
    print(f"Processed {len(data)} tweets.")
    if data:
        print(f"Sample: {data[0]} (Label: {labels[0]})")
