import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer

class DynamicSearchOptimizer:
    def __init__(self, kmeans_model_path="models/kmeans_model.pkl", embeddings_path="data/train_embeddings.pkl", sbert_model_name='all-MiniLM-L6-v2'):
        if not os.path.exists(kmeans_model_path):
            raise FileNotFoundError("Clustering model not found. Run clusterer.py first.")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError("Embeddings not found. Run embeddings.py first.")
            
        with open(kmeans_model_path, "rb") as f:
            saved_data = pickle.load(f)
            self.kmeans = saved_data["model"]
            self.mapping = saved_data["mapping"]
            
        with open(embeddings_path, "rb") as f:
            emb_data = pickle.load(f)
            self.all_embeddings = emb_data["embeddings"]
            self.all_labels = emb_data["labels"]
            
        self.sbert = SentenceTransformer(sbert_model_name)
        self.topic_to_id = {"Health": 0, "Sports": 1, "Politics": 2}
        
    def predict_cluster(self, prompt):
        """
        Embeds the prompt and predicts which topic ID it belongs to.
        """
        embedding = self.sbert.encode([prompt])
        raw_cluster_id = self.kmeans.predict(embedding)[0]
        topic_name = self.mapping[raw_cluster_id]
        return self.topic_to_id[topic_name], topic_name, embedding[0]

    def simulated_search(self, prompt_embedding, target_embeddings):
        """
        Simulates a semantic search by calculating cosine similarity.
        Target embeddings can be the full set or a cluster subset.
        """
        import time
        start = time.time()
        # Cosine similarity calculation (brute force for simulation)
        norm_a = np.linalg.norm(prompt_embedding)
        norm_b = np.linalg.norm(target_embeddings, axis=1)
        similarities = np.dot(target_embeddings, prompt_embedding) / (norm_a * norm_b + 1e-9)
        # Fake 'processing' each result to simulate real LLM work
        for _ in range(len(target_embeddings)):
            pass 
        return time.time() - start


if __name__ == '__main__':
    optimizer = DynamicSearchOptimizer()
    
    test_prompts = [
        "What are the symptoms of the flu?",
        "Who won the championship match last night?",
        "The recent election results were surprising."
    ]
    
    for p in test_prompts:
        cid = optimizer.reduce_search_space(p)
        print("-" * 20)
