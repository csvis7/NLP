import praw
import os
import time

def get_reddit_instance():
    # In a real production app, these should be env vars.
    # Hardcoded here per user request for this specific task context.
    reddit = praw.Reddit(
        client_id="ryZkJGgYx2wsH3cIYm6BDA",
        client_secret="t9-4UKGaybDwWtyk49zFewuHVSQEyg",
        user_agent="python:nlp_scraper:v1.0 (by /u/user)"
    )
    return reddit

def fetch_reddit_posts_by_word_count(topic, target_word_count=25000):
    """
    Fetches posts from Reddit until the total word count exceeds target_word_count.
    """
    print(f"Starting scrape for '{topic}' (Target: {target_word_count} words)...")
    reddit = get_reddit_instance()
    subreddit = reddit.subreddit("all")
    
    unique_ids = set()
    collected_texts = []
    current_word_count = 0
    
    # Sorts to iterate through
    sorts = ['relevance', 'hot', 'top', 'new']
    time_filters = ['all', 'year', 'month']
    
    for sort in sorts:
        if current_word_count >= target_word_count:
            break
            
        print(f"  Searching -> Sort: {sort}")
        
        if sort == 'top':
            current_times = time_filters
        else:
            current_times = [None]
            
        for tf in current_times:
            if current_word_count >= target_word_count:
                break
            
            tf_label = f" (Time: {tf})" if tf else ""
            print(f"    Fetching{tf_label}...")
            
            try:
                if sort == 'top':
                    iterator = subreddit.search(topic, sort=sort, time_filter=tf, limit=1000)
                else:
                    iterator = subreddit.search(topic, sort=sort, limit=1000)
                
                count_this_batch = 0
                words_this_batch = 0
                
                for post in iterator:
                    if current_word_count >= target_word_count:
                        break
                        
                    if post.id in unique_ids:
                        continue
                        
                    unique_ids.add(post.id)
                    
                    title = post.title
                    selftext = post.selftext
                    content = f"{title}\n{selftext}".strip()
                    
                    if content:
                        word_count = len(content.split())
                        collected_texts.append(content)
                        current_word_count += word_count
                        
                        count_this_batch += 1
                        words_this_batch += word_count
                
                print(f"      + Got {count_this_batch} posts ({words_this_batch} words). Total: {current_word_count}/{target_word_count} words")
                time.sleep(1)
                
            except Exception as e:
                print(f"      Error fetching batch: {e}")
                time.sleep(2)
                
    print(f"Scrape finished. Collected {len(collected_texts)} posts totaling {current_word_count} words for {topic}.")
    return collected_texts

def save_dataset(all_posts, directory="data"):
    """
    Saves all collected posts into a single labeled CSV file.
    Each row: text (cleaned post), label (topic name)
    """
    import csv
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, "dataset.csv")
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        for text, label in all_posts:
            clean_text = text.replace("\n", " ").strip()
            if clean_text:
                writer.writerow({"text": clean_text, "label": label})

    total = len(all_posts)
    print(f"Saved {total} total posts to {file_path}")

if __name__ == "__main__":
    topics = ["Health", "Sports", "Politics"]
    target_words = 25000

    all_posts = []   # list of (text, label) tuples
    for topic in topics:
        posts = fetch_reddit_posts_by_word_count(topic, target_word_count=target_words)
        if posts:
            for post in posts:
                all_posts.append((post, topic))
            print(f"Collected {len(posts)} posts for {topic}")
        else:
            print(f"No data found for {topic}.")

    if all_posts:
        save_dataset(all_posts)
