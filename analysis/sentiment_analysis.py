import os
import logging
import praw
import pandas as pd
from textblob import TextBlob
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizer, RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
import torch
import re
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn


# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RedditSentimentAnalysis")

# Configure Reddit API
reddit = praw.Reddit(client_id='HB9qSD1typ7tlkF78M7dPA',
                     client_secret='rBrpqwT2myyeNBMsUlmuZnXziNoROg',
                     user_agent='MyRedditBot/0.1 by abhinav')

def preprocess_text(text):
    """Preprocess the text by removing special characters and URLs"""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase and strip extra spaces
    return text

def fetch_reddit_posts(subreddit_name, limit=100):
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts = subreddit.hot(limit=limit)
        post_list = [
            {"text": preprocess_text(post.title + " " + post.selftext)} 
            for post in posts if post.selftext or post.title
        ]
        return post_list
    except Exception as e:
        logger.error(f"Error fetching posts: {e}")
        return []

def label_data(posts):
    """Label data using TextBlob for sentiment analysis (0 = positive, 1 = neutral, 2 = negative)"""
    labeled_data = []
    for post in posts:
        analysis = TextBlob(post["text"])
        if analysis.sentiment.polarity > 0:
            label = 0  # Positive
        elif analysis.sentiment.polarity == 0:
            label = 1  # Neutral
        else:
            label = 2  # Negative
        post["label"] = label
        labeled_data.append(post)
    return labeled_data

def load_model(model_name="distilbert"):
    """Load specified model and tokenizer"""
    if model_name == "distilbert":
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3).to(device)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    elif model_name == "roberta":
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3).to(device)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        raise ValueError("Model not recognized.")
    return model, tokenizer

def fine_tune_bert(posts, model_name="distilbert"):
    """Fine-tune BERT model on sentiment-labeled posts"""
    df = pd.DataFrame(posts)
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Convert to Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    model, tokenizer = load_model(model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=True, truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy='steps',
        save_steps=100,
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()
    trainer.save_model('./results')
    tokenizer.save_pretrained('./results')
    return model, tokenizer

def analyze_sentiments(posts, model, tokenizer):
    """Analyze sentiments of posts using the fine-tuned BERT model"""
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    sentiment_scores = {"positive": 0, "neutral": 0, "negative": 0}

    label_map = {0: "positive", 1: "neutral", 2: "negative"}

    for post in posts:
        sentiment = sentiment_analyzer(post["text"][:512])[0]["label"]
        if sentiment == 'LABEL_0':
            sentiment_label = "positive"
        elif sentiment == 'LABEL_1':
            sentiment_label = "neutral"
        else:
            sentiment_label = "negative"
        sentiment_scores[sentiment_label] += 1

    return sentiment_scores

def display_results(sentiment_scores, method_name):
    """Display sentiment analysis results"""
    total = sum(sentiment_scores.values())
    if total == 0:
        logger.info(f"{method_name} Sentiment Analysis Results: No posts analyzed.")
        return

    logger.info(f"{method_name} Sentiment Analysis Results:")
    for sentiment, count in sentiment_scores.items():
        percentage = (count / total) * 100
        logger.info(f"{sentiment.capitalize()}: {percentage:.2f}%")

def cross_validate_bert(posts, model_name="distilbert"):
    """Perform cross-validation for sentiment classification"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for train_index, val_index in kf.split(posts):
        train_posts = [posts[i] for i in train_index]
        val_posts = [posts[i] for i in val_index]

        # Fine-tune on train_posts
        model, tokenizer = fine_tune_bert(train_posts, model_name)

        # Analyze sentiments on val_posts
        sentiment_scores = analyze_sentiments(val_posts, model, tokenizer)
        results.append(sentiment_scores)
        
    # Average results
    avg_results = {key: sum([r[key] for r in results]) / len(results) for key in results[0]}
    return avg_results

def evaluate_model(true_labels, predictions):
    """Evaluate model performance using classification report and confusion matrix"""
    report = classification_report(true_labels, predictions)
    logger.info(f"Classification Report:\n{report}")
    
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['positive', 'neutral', 'negative'], yticklabels=['positive', 'neutral', 'negative'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def run():
    subreddit_name = 'History'
    posts = fetch_reddit_posts(subreddit_name)
    logger.info(f"Fetched {len(posts)} posts from r/{subreddit_name}.")
    
    if posts:
        labeled_posts = label_data(posts)

        # Check if model already exists
        if os.path.exists('./results'):
            logger.info("Loading existing model and tokenizer...")
            model = DistilBertForSequenceClassification.from_pretrained('./results').to(device)
            tokenizer = DistilBertTokenizer.from_pretrained('./results')
        else:
            model, tokenizer = fine_tune_bert(labeled_posts)

        sentiment_scores = analyze_sentiments(labeled_posts, model, tokenizer)
        display_results(sentiment_scores, "BERT")
        
        # Optionally perform cross-validation
        avg_results = cross_validate_bert(labeled_posts, model_name="distilbert")
        logger.info(f"Average cross-validation results: {avg_results}")
    else:
        logger.warning("No posts fetched to analyze.")

if __name__ == "__main__":
    run()
