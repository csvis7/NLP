"""
evaluator.py

Loads the held-out test set (data/test.csv) and evaluates the trained K-Means
cluster model against the ground-truth labels.

Pipeline:
  1. Load data/test.csv (30% held-out split).
  2. Encode test sentences with SBERT.
  3. Predict cluster for each test sentence using kmeans_model.
  4. Translate cluster IDs → topic names using the saved mapping.
  5. Compare predictions against true labels.
  6. Report: Accuracy, Confusion Matrix, Homogeneity, Completeness, V-Measure.
"""

import csv
import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    adjusted_rand_score,
    classification_report,
)


def load_test_data(test_path="data/test.csv"):
    texts, labels = [], []
    with open(test_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["text"].strip():
                texts.append(row["text"].strip())
                labels.append(row["label"].strip())
    return texts, labels


def predict_topics(texts, kmeans_model, label_mapping, model_name="all-MiniLM-L6-v2"):
    """
    Encodes texts with SBERT and predicts topic name for each via K-Means.
    Returns (predicted_topics list[str], embeddings ndarray).
    """
    print(f"Encoding {len(texts)} test sentences...")
    sbert = SentenceTransformer(model_name)
    embeddings = sbert.encode(texts, convert_to_numpy=True)

    cluster_ids = kmeans_model.predict(embeddings)
    predicted_topics = [label_mapping[cid] for cid in cluster_ids]
    return predicted_topics, embeddings


def run_evaluation(true_labels, predicted_labels):
    """
    Prints a comprehensive evaluation report.
    """
    unique_topics = sorted(set(true_labels))

    print("\n" + "=" * 50)
    print("       TEST SET EVALUATION REPORT")
    print("=" * 50)

    # --- Overall Accuracy ---
    acc = accuracy_score(true_labels, predicted_labels)
    print(f"\nOverall Accuracy:   {acc:.4f} ({acc*100:.1f}%)")

    # --- Supervised Metrics ---
    print("\n--- Supervised Metrics (True Labels vs Predictions) ---")
    print(f"  Homogeneity:       {homogeneity_score(true_labels, predicted_labels):.4f}")
    print(f"  Completeness:      {completeness_score(true_labels, predicted_labels):.4f}")
    print(f"  V-Measure:         {v_measure_score(true_labels, predicted_labels):.4f}")
    print(f"  Adj. Rand Index:   {adjusted_rand_score(true_labels, predicted_labels):.4f}")

    # --- Per-class Report ---
    print("\n--- Per-Class Report ---")
    print(classification_report(true_labels, predicted_labels, target_names=unique_topics))

    # --- Confusion Matrix ---
    print("--- Confusion Matrix ---")
    print("  Rows = True Label | Columns = Predicted Label\n")
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_topics)
    cm_df = pd.DataFrame(cm, index=unique_topics, columns=unique_topics)
    print(cm_df)

    # --- Majority-cluster validation ---
    print("\n--- Majority Cluster Validation ---")
    for topic in unique_topics:
        topic_mask = [i for i, t in enumerate(true_labels) if t == topic]
        preds_for_topic = [predicted_labels[i] for i in topic_mask]
        from collections import Counter
        majority = Counter(preds_for_topic).most_common(1)[0]
        match = "[OK]" if majority[0] == topic else "[FAIL]"
        print(f"  {topic}: majority prediction = '{majority[0]}' "
              f"({majority[1]}/{len(preds_for_topic)}) {match}")

    print("=" * 50)
    return acc


if __name__ == "__main__":
    test_path  = "data/test.csv"
    model_path = "models/kmeans_model.pkl"

    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found. Run dataset_builder.py first.")
        exit(1)
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Run clusterer.py first.")
        exit(1)

    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    kmeans_model  = saved["model"]
    label_mapping = saved["mapping"]
    print(f"Loaded model. Cluster mapping: {label_mapping}")

    texts, true_labels = load_test_data(test_path)
    print(f"Test set: {len(texts)} samples")

    predicted_labels, _ = predict_topics(texts, kmeans_model, label_mapping)
    run_evaluation(true_labels, predicted_labels)
