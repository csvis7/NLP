from crawler import generate_mock_tweets, save_tweets
from data_processor import process_data
from embeddings import generate_word_embeddings, generate_sentence_embeddings, save_embeddings
from clusterer import perform_clustering, evaluate_clustering
from search_optimizer import DynamicSearchOptimizer
import os
import pickle

def main():
    # 1. Data Collection (Mock Crawler)
    print("--- Phase 1: Data Collection ---")
    topics = ["Health", "Sports", "Politics"]
    target_words = 25000
    for topic in topics:
        tweets = generate_mock_tweets(topic, target_words)
        save_tweets(tweets, topic)

    # 2. Data Processing
    print("\n--- Phase 2: Data Processing ---")
    data, labels = process_data()
    print(f"Total processed tweets: {len(data)}")

    # 3. Embedding Generation
    print("\n--- Phase 3: Embedding Generation ---")
    tokenized_data = [d.split() for d in data]
    w2v_model = generate_word_embeddings(tokenized_data)
    w2v_model.save("models/word2vec.model")
    
    sbert_embeddings, _ = generate_sentence_embeddings(data)
    save_embeddings(sbert_embeddings, labels)

    # 4. Clustering & Evaluation
    print("\n--- Phase 4: Clustering & Evaluation ---")
    cluster_labels, kmeans_model = perform_clustering(sbert_embeddings)
    metrics = evaluate_clustering(sbert_embeddings, cluster_labels)
    
    # Dynamically determine mapping
    from clusterer import map_clusters_to_topics
    label_mapping = map_clusters_to_topics(labels, cluster_labels)
    print(f"Cluster Mapping: {label_mapping}")

    print("\n--- PERFORMANCE METRICS ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    with open("models/kmeans_model.pkl", "wb") as f:
        pickle.dump({"model": kmeans_model, "mapping": label_mapping}, f)

    # 5. Dynamic State Search Demonstration
    print("\n--- Phase 5: Dynamic State Search (Search Space Reduction) ---")
    optimizer = DynamicSearchOptimizer()
    import time
    import numpy as np
    
    while True:
        user_prompt = input("\nEnter your prompt (or 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            break
            
        # 1. Measure Prediction Latency
        start_time = time.time()
        topic_id, topic_name, prompt_embedding = optimizer.predict_cluster(user_prompt)
        prediction_latency = time.time() - start_time
        
        # 2. Measure Traditional Search Time (Actual for 75k words)
        raw_traditional_time = optimizer.simulated_search(prompt_embedding, optimizer.all_embeddings)
        
        # 3. Measure Optimized Search Time (Actual for one cluster)
        cluster_indices = [i for i, label in enumerate(optimizer.all_labels) if label == topic_name]
        cluster_embeddings = optimizer.all_embeddings[cluster_indices]
        raw_optimized_search_time = optimizer.simulated_search(prompt_embedding, cluster_embeddings)
        
        # 4. SCALE UP FOR REAL-WORLD LLM SCENARIO (e.g., 1000x larger database)
        SCALING_FACTOR = 1000 
        scaled_traditional_time = raw_traditional_time * SCALING_FACTOR
        scaled_optimized_search_time = raw_optimized_search_time * SCALING_FACTOR
        total_optimized_time = prediction_latency + scaled_optimized_search_time
        
        print("\n" + "="*50)
        print(f"   DYNAMIC STATE SEARCH PERFORMANCE (X{SCALING_FACTOR} SCALE)")
        print("="*50)
        print(f"User Prompt: '{user_prompt}'")
        print(f"Predicted Topic: {topic_name} (ID: {topic_id})")
        print("-" * 50)
        print(f"1. TRADITIONAL LLM SEARCH:         {scaled_traditional_time:.6f}s")
        print(f"2. DYNAMIC STATE SEARCH (MODEL):")
        print(f"   - Reduction Latency (Fixed):    {prediction_latency:.6f}s")
        print(f"   - Scaled Reduced Search:        {scaled_optimized_search_time:.6f}s")
        print(f"   - TOTAL OPTIMIZED TIME:         {total_optimized_time:.6f}s")
        print("-" * 50)
        
        time_saved = scaled_traditional_time - total_optimized_time
        improvement = (time_saved / scaled_traditional_time) * 100
        print(f"TIME SAVED: {time_saved:.6f}s ({improvement:.1f}% Improvement)")
        print("="*50)
        print(f"Note: Scaled by {SCALING_FACTOR}x to simulate production search space.")







if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
    main()
