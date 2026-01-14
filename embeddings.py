from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle

def generate_word_embeddings(tokenized_sentences, vector_size=100, window=5, min_count=1):
    """
    Generates Word2Vec word embeddings.
    """
    print("Training Word2Vec model...")
    model = Word2Vec(sentences=tokenized_sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

def generate_sentence_embeddings(sentences, model_name='all-MiniLM-L6-v2'):
    """
    Generates sentence embeddings using SBERT.
    """
    print(f"Loading SBERT model ({model_name})...")
    model = SentenceTransformer(model_name)
    print("Encoding sentences...")
    embeddings = model.encode(sentences)
    return embeddings, model

def save_embeddings(embeddings, labels, file_path="data/embeddings.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels}, f)
    print(f"Embeddings saved to {file_path}")

if __name__ == "__main__":
    from data_processor import process_data
    
    data, labels = process_data()
    if not data:
        print("No data found. Run crawler.py first.")
    else:
        # Word2Vec requires tokenized sentences
        tokenized_data = [d.split() for d in data]
        w2v_model = generate_word_embeddings(tokenized_data)
        print(f"Word2Vec vocab size: {len(w2v_model.wv)}")
        
        # SBERT embeddings
        sbert_embeddings, sbert_model = generate_sentence_embeddings(data)
        save_embeddings(sbert_embeddings, labels)
        
        # Save models for later use
        if not os.path.exists("models"):
            os.makedirs("models")
        w2v_model.save("models/word2vec.model")
        # SBERT model can be reloaded by name, no need to save local weights unless fine-tuned
