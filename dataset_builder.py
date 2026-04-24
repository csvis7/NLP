"""
dataset_builder.py

Reads data/dataset.csv and produces:
  - data/train.csv          (70% stratified split)
  - data/test.csv           (30% stratified split)
  - data/query_vectors.pkl  (SBERT-encoded representative queries per topic)
"""

import os
import csv
import pickle
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Representative query phrases per topic.
# These are hand-crafted short phrases that strongly represent each topic.
# SBERT will encode these into vectors and their mean becomes the
# "anchor vector" used to label K-Means clusters by minimum cosine distance.
# ---------------------------------------------------------------------------
QUERY_DICT = {
    "Sports": [
        "football match result",
        "player scored a goal",
        "championship final winner",
        "basketball game score",
        "olympic athlete performance",
        "cricket world cup",
        "tennis grand slam tournament",
        "sports team ranking",
    ],
    "Health": [
        "vaccine dose immune response",
        "hospital patient treatment",
        "mental health awareness",
        "nutrition diet fitness",
        "disease symptoms medication",
        "medical research clinical trial",
        "public health policy",
        "doctor nurse healthcare",
    ],
    "Politics": [
        "senate vote bill congress",
        "election campaign result",
        "government policy reform",
        "president prime minister decision",
        "political party debate",
        "international diplomacy treaty",
        "parliament legislation",
        "democracy rights protest",
    ],
}


def stratified_split(dataset_path="data/dataset.csv", train_ratio=0.7, seed=42):
    """
    Reads dataset.csv and returns two lists: (train_rows, test_rows).
    Split is stratified — each topic gets the same train/test ratio.
    """
    np.random.seed(seed)
    # group rows by label
    buckets = defaultdict(list)
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            buckets[row["label"]].append(row)

    train_rows, test_rows = [], []
    for label, rows in buckets.items():
        np.random.shuffle(rows)
        split_idx = int(len(rows) * train_ratio)
        train_rows.extend(rows[:split_idx])
        test_rows.extend(rows[split_idx:])
        print(f"  {label}: {split_idx} train / {len(rows) - split_idx} test")

    return train_rows, test_rows


def save_split(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows → {path}")


def build_query_vectors(model_name="all-MiniLM-L6-v2"):
    """
    Encodes representative queries for each topic using SBERT.
    Returns a dict: { topic_name: mean_vector (np.ndarray, shape [384]) }
    """
    print(f"Loading SBERT ({model_name}) for query encoding...")
    model = SentenceTransformer(model_name)
    query_vectors = {}
    for topic, phrases in QUERY_DICT.items():
        vecs = model.encode(phrases, convert_to_numpy=True)   # shape [n, 384]
        query_vectors[topic] = vecs.mean(axis=0)              # shape [384]
        print(f"  {topic}: encoded {len(phrases)} queries → mean vector shape {query_vectors[topic].shape}")
    return query_vectors


if __name__ == "__main__":
    dataset_path = "data/dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"ERROR: {dataset_path} not found. Run crawler.py first.")
        exit(1)

    print("\n--- Stratified Train/Test Split (70/30) ---")
    train_rows, test_rows = stratified_split(dataset_path)
    save_split(train_rows, "data/train.csv")
    save_split(test_rows,  "data/test.csv")

    print("\n--- Building Query Vector Dictionary ---")
    query_vectors = build_query_vectors()

    os.makedirs("data", exist_ok=True)
    with open("data/query_vectors.pkl", "wb") as f:
        pickle.dump(query_vectors, f)
    print("Query vectors saved → data/query_vectors.pkl")

    print("\nDataset build complete.")
    print(f"  Total train: {len(train_rows)}")
    print(f"  Total test:  {len(test_rows)}")
