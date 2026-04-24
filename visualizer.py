import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

def visualize_clusters(embeddings_path="data/train_embeddings.pkl", model_path="models/kmeans_model.pkl", output_file="cluster_plot.png"):
    if not os.path.exists(embeddings_path) or not os.path.exists(model_path):
        print("Required data files not found.")
        return

    # Load data
    print("Loading data...")
    with open(embeddings_path, "rb") as f:
        data = pickle.load(f)
        embeddings = data["embeddings"]
        # Use predicted labels from the model to align with mapping
        
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        kmeans = model_data["model"]
        mapping = model_data["mapping"]
        
    # Get cluster labels
    cluster_labels = kmeans.predict(embeddings)
    
    # Map cluster IDs to Topic Names
    topic_labels = [mapping[label] for label in cluster_labels]
    
    # Reduce dimensions using t-SNE (better for visualizing clusters than PCA)
    print("Reducing dimensions (t-SNE)...")
    from sklearn.manifold import TSNE
    
    # t-SNE for 2D projection
    tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity=40, n_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Plotting
    print("Plotting...")
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=reduced_embeddings[:, 0], 
        y=reduced_embeddings[:, 1], 
        hue=topic_labels, 
        palette="viridis",
        s=80,
        alpha=0.6,
        edgecolor="w", 
        linewidth=0.5
    )
    
    plt.title("Cluster Visualization (t-SNE Projection)", fontsize=18)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Topic", title_fontsize=12, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.15)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    visualize_clusters()
