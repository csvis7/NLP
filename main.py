"""
main.py — NLP Pipeline Orchestrator

Pipeline order:
  1. [Optional] crawler.py         → data/dataset.csv
  2. dataset_builder.py            → data/train.csv, data/test.csv, data/query_vectors.pkl
  3. embeddings.py                 → data/train_embeddings.pkl
  4. clusterer.py                  → models/kmeans_model.pkl  (distance-based labeling)
  5. evaluator.py                  → accuracy + confusion matrix on test set
  6. visualizer.py                 → cluster_plot.png (t-SNE)
  7. [Interactive] semantic router → dynamic state search demo
"""

import os
import pickle
import time
import numpy as np

def main():
    # -------------------------------------------------------------------------
    # Phase 1: Data Collection
    # -------------------------------------------------------------------------
    print("--- Phase 1: Data Collection ---")
    if os.path.exists("data/dataset.csv"):
        print("Skipping: data/dataset.csv already exists.")
    else:
        from crawler import fetch_reddit_posts_by_word_count, save_dataset
        topics = ["Health", "Sports", "Politics"]
        all_posts = []
        for topic in topics:
            posts = fetch_reddit_posts_by_word_count(topic, target_word_count=25000)
            for post in posts:
                all_posts.append((post, topic))
        save_dataset(all_posts)

    # -------------------------------------------------------------------------
    # Phase 2: Dataset Building (split + query vectors)
    # -------------------------------------------------------------------------
    print("\n--- Phase 2: Dataset Building (70/30 Split + Query Vectors) ---")
    if os.path.exists("data/train.csv") and os.path.exists("data/query_vectors.pkl"):
        print("Skipping: train.csv and query_vectors.pkl already exist.")
    else:
        from dataset_builder import stratified_split, save_split, build_query_vectors
        train_rows, test_rows = stratified_split("data/dataset.csv")
        save_split(train_rows, "data/train.csv")
        save_split(test_rows,  "data/test.csv")
        query_vectors = build_query_vectors()
        with open("data/query_vectors.pkl", "wb") as f:
            pickle.dump(query_vectors, f)
        print("Query vectors saved → data/query_vectors.pkl")

    # -------------------------------------------------------------------------
    # Phase 3: Embedding Generation (train set only)
    # -------------------------------------------------------------------------
    print("\n--- Phase 3: Embedding Generation (Train Set) ---")
    if os.path.exists("data/train_embeddings.pkl"):
        print("Skipping: train_embeddings.pkl already exists.")
        with open("data/train_embeddings.pkl", "rb") as f:
            emb_data = pickle.load(f)
        train_embeddings = emb_data["embeddings"]
        train_labels = emb_data["labels"]
        print(f"Loaded {len(train_labels)} training embeddings.")
    else:
        from embeddings import load_train_data, generate_sentence_embeddings, save_embeddings
        texts, train_labels = load_train_data()
        train_embeddings, _ = generate_sentence_embeddings(texts)
        save_embeddings(train_embeddings, train_labels)

    # -------------------------------------------------------------------------
    # Phase 4: Clustering & Distance-Based Cluster Labeling
    # -------------------------------------------------------------------------
    print("\n--- Phase 4: Clustering & Cluster Labeling ---")
    from clusterer import (
        perform_clustering, assign_clusters_by_query_distance, evaluate_clustering
    )

    with open("data/query_vectors.pkl", "rb") as f:
        query_vectors = pickle.load(f)

    cluster_labels, kmeans_model = perform_clustering(train_embeddings)
    label_mapping = assign_clusters_by_query_distance(
        kmeans_model, train_embeddings, cluster_labels, query_vectors
    )
    print(f"Final Mapping: {label_mapping}")

    metrics = evaluate_clustering(train_embeddings, cluster_labels)
    print("\n--- Clustering Metrics (Train) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs("models", exist_ok=True)
    with open("models/kmeans_model.pkl", "wb") as f:
        pickle.dump({"model": kmeans_model, "mapping": label_mapping}, f)
    print("Model saved → models/kmeans_model.pkl")

    # -------------------------------------------------------------------------
    # Phase 5: Evaluation on Held-Out Test Set
    # -------------------------------------------------------------------------
    print("\n--- Phase 5: Test Set Evaluation ---")
    if not os.path.exists("data/test.csv"):
        print("WARNING: data/test.csv not found, skipping evaluation.")
    else:
        from evaluator import load_test_data, predict_topics, run_evaluation
        test_texts, test_true_labels = load_test_data()
        predicted_labels, _ = predict_topics(test_texts, kmeans_model, label_mapping)
        run_evaluation(test_true_labels, predicted_labels)

    # -------------------------------------------------------------------------
    # Phase 6: Visualization (t-SNE)
    # -------------------------------------------------------------------------
    print("\n--- Phase 6: Visualization (t-SNE) ---")
    try:
        from visualizer import visualize_clusters
        visualize_clusters()
    except Exception as e:
        print(f"Visualization skipped: {e}")

    # -------------------------------------------------------------------------
    # Phase 7: Interactive Semantic Router Demo
    # -------------------------------------------------------------------------
    print("\n--- Phase 7: Dynamic State Search Demo ---")
    try:
        from search_optimizer import DynamicSearchOptimizer
        optimizer = DynamicSearchOptimizer()
        SCALE = 1000

        while True:
            user_prompt = input("\nEnter your prompt (or 'exit' to quit): ")
            if user_prompt.lower() == "exit":
                break

            start = time.time()
            topic_id, topic_name, prompt_embedding = optimizer.predict_cluster(user_prompt)
            pred_latency = time.time() - start

            raw_trad  = optimizer.simulated_search(prompt_embedding, optimizer.all_embeddings)
            cluster_idx = [i for i, l in enumerate(optimizer.all_labels) if l == topic_name]
            raw_opt   = optimizer.simulated_search(
                prompt_embedding, optimizer.all_embeddings[cluster_idx]
            )

            scaled_trad  = raw_trad  * SCALE
            scaled_opt   = raw_opt   * SCALE
            total_opt    = pred_latency + scaled_opt
            saved        = scaled_trad - total_opt

            print(f"\n{'='*50}")
            print(f"   DYNAMIC STATE SEARCH (x{SCALE} SCALE)")
            print(f"{'='*50}")
            print(f"  Prompt:             {user_prompt}")
            print(f"  Predicted Topic:    {topic_name} (ID: {topic_id})")
            print(f"  Traditional Search: {scaled_trad:.6f}s")
            print(f"  Optimized Search:   {total_opt:.6f}s  "
                  f"(latency {pred_latency:.4f}s + search {scaled_opt:.4f}s)")
            print(f"  Time Saved:         {saved:.6f}s  ({saved/scaled_trad*100:.1f}% faster)")
            print(f"{'='*50}")
    except Exception as e:
        print(f"Search demo unavailable: {e}")


if __name__ == "__main__":
    main()
