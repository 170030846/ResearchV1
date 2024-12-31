import praw
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud

# Replace these with your own credentials
reddit = praw.Reddit(client_id='HB9qSD1typ7tlkF78M7dPA',
                     client_secret='rBrpqwT2myyeNBMsUlmuZnXziNoROg',
                     user_agent='MyRedditBot/0.1 by abhinav')

def fetch_reddit_posts(subreddit_name, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = subreddit.hot(limit=limit)
    post_data = []
    for post in posts:
        post_data.append({
            'title': post.title,
            'score': post.score,  # Upvotes - Downvotes
            'num_comments': post.num_comments,
            'url': post.url,
            'created_utc': post.created_utc  # Add the post creation time in UTC
        })
    return post_data

def calculate_engagement_growth(posts):
    current_time = datetime.utcnow().timestamp()
    engagement_growth = []

    for post in posts:
        # Calculate the time difference in hours from when the post was created
        age_in_hours = (current_time - post['created_utc']) / 3600
        if age_in_hours == 0:
            age_in_hours = 1  # Avoid division by zero

        # Calculate the engagement growth rate (increase in score per hour)
        engagement_growth_rate = post['score'] / age_in_hours
        engagement_growth.append({
            'title': post['title'],
            'engagement_growth_rate': engagement_growth_rate,
            'url': post['url']
        })

    return sorted(engagement_growth, key=lambda x: x['engagement_growth_rate'], reverse=True)

def advance_calculate_engagement(posts):
    engagement_scores = []
    current_time = datetime.utcnow().timestamp()

    for post in posts:
        age_in_hours = (current_time - post['created_utc']) / 3600  # Age in hours
        if age_in_hours == 0:
            age_in_hours = 1  # Avoid division by zero

        comment_upvote_ratio = post['num_comments'] / post['score'] if post['score'] != 0 else 0
        normalized_score = post['score'] / age_in_hours  # Normalize score based on age

        engagement_score = (normalized_score + comment_upvote_ratio)  # Sophisticated engagement metric
        engagement_scores.append({
            'title': post['title'],
            'engagement_score': engagement_score,
            'url': post['url']
        })
    return sorted(engagement_scores, key=lambda x: x['engagement_score'], reverse=True)

def plot_engagement_time_series(posts):
    times = [datetime.utcfromtimestamp(post['created_utc']) for post in posts]
    engagement_scores = [post['score'] + post['num_comments'] for post in posts]

    # Plot engagement over time
    plt.figure(figsize=(10, 6))
    plt.plot(times, engagement_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Time')
    plt.ylabel('Engagement (Score + Comments)')
    plt.title('Engagement Time Series')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_engagement_heatmap(posts):
    # Get the post creation hour and engagement
    times = [datetime.utcfromtimestamp(post['created_utc']).hour for post in posts]
    engagement_scores = [post['score'] + post['num_comments'] for post in posts]

    # Create a DataFrame
    data = pd.DataFrame({'hour': times, 'engagement_score': engagement_scores})

    # Pivot data for the heatmap
    heatmap_data = data.groupby('hour').sum().reset_index()

    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data.pivot_table(index='hour', values='engagement_score', aggfunc='sum').T, annot=True, cmap='YlGnBu')
    plt.title('Engagement Heatmap by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Engagement Score')
    plt.show()

def generate_word_cloud(posts):
    titles = [post['title'] for post in posts]
    text = ' '.join(titles)
    
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Display the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def fetch_multiple_subreddits(subreddit_names, limit=100):
    all_posts = []
    for name in subreddit_names:
        posts = fetch_reddit_posts(name, limit)
        all_posts.extend(posts)
    return all_posts

def run():
    subreddit_name = 'History'
    posts = fetch_reddit_posts(subreddit_name)
    print(f"Fetched {len(posts)} posts from r/{subreddit_name}.")

    engagement_scores = advance_calculate_engagement(posts)
    print(f"\nTop Engaging Posts:")

    for idx, post in enumerate(engagement_scores[:10], start=1):
        print(f"{idx}. {post['title']} (Engagement Score: {post['engagement_score']:.2f}) - {post['url']}")

    # Trending Post Analysis
    growth_posts = calculate_engagement_growth(posts)
    print("\nTop Trending Posts:")
    for idx, post in enumerate(growth_posts[:10], start=1):
        print(f"{idx}. {post['title']} (Growth Rate: {post['engagement_growth_rate']:.2f}) - {post['url']}")

    # Time Series Analysis of Engagement
    plot_engagement_time_series(posts)

    # Heatmap of Engagement by Hour
    plot_engagement_heatmap(posts)

    # Generate a Word Cloud from Post Titles
    generate_word_cloud(engagement_scores[:50])  # Use top 50 posts for word cloud

    # Comparison Between Subreddits
    subreddits = ['History', 'Science', 'Technology']
    all_posts = fetch_multiple_subreddits(subreddits)
    engagement_scores = advance_calculate_engagement(all_posts)

    print("\nTop Engaging Posts Across Multiple Subreddits:")
    for idx, post in enumerate(engagement_scores[:10], start=1):
        print(f"{idx}. {post['title']} (Engagement Score: {post['engagement_score']:.2f}) - {post['url']}")

if __name__ == "__main__":
    run()
