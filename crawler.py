import random
import os

def generate_mock_tweets(topic, num_words_target):
    """
    Simulates crawling tweets for a given topic until a word count target is reached.
    """
    vocabulary = {
        "health": ["vaccine", "exercise", "nutrition", "mental health", "doctor", "hospital", "wellness", "diet", "flu", "symptoms", "therapy", "medicine", "immunity", "patient", "clinical"],
        "sports": ["football", "basketball", "olympics", "match", "goal", "tournament", "athlete", "coach", "score", "championship", "stadium", "victory", "tennis", "league", "training"],
        "politics": ["election", "government", "policy", "senate", "congress", "president", "vote", "democracy", "debate", "minister", "legislation", "campaign", "sanctions", "reform", "treaty"]
    }
    
    fillers = ["the", "is", "a", "of", "to", "and", "in", "it", "with", "on", "for", "at", "by", "that", "this"]
    
    tweets = []
    total_words = 0
    
    while total_words < num_words_target:
        tweet_len = random.randint(10, 30)
        tweet_words = []
        for _ in range(tweet_len):
            if random.random() > 0.4:
                tweet_words.append(random.choice(vocabulary[topic.lower()]))
            else:
                tweet_words.append(random.choice(fillers))
        
        tweet = " ".join(tweet_words)
        tweets.append(tweet)
        total_words += len(tweet_words)
        
    return tweets

def save_tweets(tweets, topic, directory="data"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, f"{topic.lower()}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        for tweet in tweets:
            f.write(tweet + "\n")
    print(f"Saved {len(tweets)} tweets for {topic} to {file_path}")

if __name__ == "__main__":
    topics = ["Health", "Sports", "Politics"]
    target_words = 25000
    
    for topic in topics:
        print(f"Collecting data for {topic}...")
        tweets = generate_mock_tweets(topic, target_words)
        save_tweets(tweets, topic)
