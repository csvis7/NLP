from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pickle
import numpy as np
import os

def perform_clustering(embeddings, n_clusters=3):
    """
    Groups sentence embeddings using K-Means.
    """
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels, kmeans

def evaluate_clustering(embeddings, labels):
    """
    Calculates various clustering performance metrics.
    """
    print("Evaluating clustering performance...")
    silhouette = silhouette_score(embeddings, labels)
    db_index = davies_bouldin_score(embeddings, labels)
    ch_score = calinski_harabasz_score(embeddings, labels)
    
    metrics = {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": db_index,
        "Calinski-Harabasz Score": ch_score
    }
    return metrics

def map_clusters_to_topics(true_labels, cluster_labels):
    """
    Maps cluster IDs to the original topic order: Health: 0, Sports: 1, Politics: 2.
    """
    from collections import Counter
    topic_map = {"Health": 0, "Sports": 1, "Politics": 2}
    unique_clusters = np.unique(cluster_labels)
    
    # cluster_id -> topic_name (majority vote)
    mapping = {}
    for cluster in unique_clusters:
        indices = np.where(cluster_labels == cluster)
        labels_in_cluster = [true_labels[i] for i in indices[0]]
        majority_topic = Counter(labels_in_cluster).most_common(1)[0][0]
        mapping[cluster] = majority_topic
    
    return mapping

if __name__ == "__main__":
    if not os.path.exists("data/embeddings.pkl"):
        print("Embeddings not found. Run embeddings.py first.")
    else:
        with open("data/embeddings.pkl", "rb") as f:
            data = pickle.load(f)
            embeddings = data["embeddings"]
            true_labels = data["labels"]
            
        cluster_labels, kmeans_model = perform_clustering(embeddings)
        metrics = evaluate_clustering(embeddings, cluster_labels)
        
        # Create mapping
        label_mapping = map_clusters_to_topics(true_labels, cluster_labels)
        print(f"Cluster Mapping: {label_mapping}")
        
        print("\n--- Clustering Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Save model and mapping
        if not os.path.exists("models"):
            os.makedirs("models")
        with open("models/kmeans_model.pkl", "wb") as f:
            pickle.dump({"model": kmeans_model, "mapping": label_mapping}, f)
        print("Clustering model and mapping saved to models/kmeans_model.pkl")

