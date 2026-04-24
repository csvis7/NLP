"""
clusterer.py

Steps:
  1. Load SBERT embeddings from data/train_embeddings.pkl.
  2. Run K-Means (n=3) on the embeddings.
  3. Compute each cluster's centroid.
  4. Load query_vectors.pkl and assign each cluster to the nearest topic
     using cosine distance (true unsupervised — no peeking at true labels).
  5. Save model + mapping to models/kmeans_model.pkl.
"""

import pickle
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def cosine_distance(vec_a, vec_b):
    """1 - cosine_similarity. Lower = more similar."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - np.dot(vec_a, vec_b) / (norm_a * norm_b)


def perform_clustering(embeddings, n_clusters=3):
    """K-Means clustering. Returns (cluster_labels, fitted_kmeans)."""
    print(f"Running K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels, kmeans


def assign_clusters_by_query_distance(kmeans, embeddings, cluster_labels, query_vectors):
    """
    True unsupervised cluster labeling.

    For each cluster:
      1. Compute the centroid (mean of all embedding vectors in that cluster).
      2. Measure cosine distance from the centroid to each topic's query vector.
      3. Assign the cluster to the topic with the minimum distance.

    Returns: dict { cluster_id (int) -> topic_name (str) }
    """
    unique_clusters = np.unique(cluster_labels)
    mapping = {}

    print("\n--- Cluster → Topic Assignment (Cosine Distance) ---")
    for cluster_id in unique_clusters:
        # Get all embeddings in this cluster
        mask = cluster_labels == cluster_id
        cluster_vecs = embeddings[mask]
        centroid = cluster_vecs.mean(axis=0)

        # Compute cosine distance to every topic's query vector
        distances = {}
        for topic, query_vec in query_vectors.items():
            distances[topic] = cosine_distance(centroid, query_vec)

        # The topic with the smallest distance wins
        # this is where we assign the cluster to a topic
    
        assigned_topic = min(distances, key=distances.get)
        mapping[cluster_id] = assigned_topic

        # Print breakdown for transparency
        dist_str = ", ".join(f"{t}: {d:.4f}" for t, d in distances.items())
        print(f"  Cluster {cluster_id} → {assigned_topic}  [{dist_str}]")

    return mapping


def evaluate_clustering(embeddings, cluster_labels):
    """Standard unsupervised clustering quality metrics."""
    print("\nEvaluating clustering performance...")
    return {
        "Silhouette Score":        silhouette_score(embeddings, cluster_labels),
        "Davies-Bouldin Index":    davies_bouldin_score(embeddings, cluster_labels),
        "Calinski-Harabasz Score": calinski_harabasz_score(embeddings, cluster_labels),
    }


if __name__ == "__main__":
    # Load training embeddings
    emb_path = "data/train_embeddings.pkl"
    qv_path  = "data/query_vectors.pkl"

    if not os.path.exists(emb_path):
        print(f"ERROR: {emb_path} not found. Run embeddings.py first.")
        exit(1)
    if not os.path.exists(qv_path):
        print(f"ERROR: {qv_path} not found. Run dataset_builder.py first.")
        exit(1)

    with open(emb_path, "rb") as f:
        data = pickle.load(f)
    embeddings    = data["embeddings"]
    true_labels   = data["labels"]        # kept for cross-checking only

    with open(qv_path, "rb") as f:
        query_vectors = pickle.load(f)

    # Cluster
    cluster_labels, kmeans_model = perform_clustering(embeddings)

    # Assign clusters → topics via cosine distance (no true labels used here)
    label_mapping = assign_clusters_by_query_distance(
        kmeans_model, embeddings, cluster_labels, query_vectors
    )
    print(f"\nFinal Mapping: {label_mapping}")

    # Evaluate
    metrics = evaluate_clustering(embeddings, cluster_labels)
    print("\n--- Clustering Metrics (Training Data) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save model + mapping
    os.makedirs("models", exist_ok=True)
    with open("models/kmeans_model.pkl", "wb") as f:
        pickle.dump({"model": kmeans_model, "mapping": label_mapping}, f)
    print("\nModel + mapping saved → models/kmeans_model.pkl")
