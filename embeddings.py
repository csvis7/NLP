"""
embeddings.py

Reads data/train.csv (70% training split) and generates SBERT sentence embeddings.
Saves to data/train_embeddings.pkl.
"""

import csv
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


def load_train_data(train_path="data/train.csv"):
    """
    Reads train.csv and returns (texts, labels) lists.
    """
    texts, labels = [], []
    with open(train_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["text"].strip():
                texts.append(row["text"].strip())
                labels.append(row["label"].strip())
    return texts, labels


def generate_sentence_embeddings(sentences, model_name="all-MiniLM-L6-v2"):
    """
    Encodes sentences using SBERT.
    Returns (embeddings ndarray [N, 384], model).
    """
    print(f"Loading SBERT model ({model_name})...")
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(sentences)} sentences...")
    embeddings = model.encode(sentences, convert_to_numpy=True)
    print(f"Embedding shape: {embeddings.shape}")
    return embeddings, model


def save_embeddings(embeddings, labels, file_path="data/train_embeddings.pkl"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels}, f)
    print(f"Embeddings saved → {file_path}")


if __name__ == "__main__":
    if not os.path.exists("data/train.csv"):
        print("ERROR: data/train.csv not found. Run dataset_builder.py first.")
        exit(1)

    texts, labels = load_train_data()
    print(f"Loaded {len(texts)} training sentences.")

    embeddings, _ = generate_sentence_embeddings(texts)
    save_embeddings(embeddings, labels)
