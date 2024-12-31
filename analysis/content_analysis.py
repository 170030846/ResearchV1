def run():
    import nltk
    import praw
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from gensim import corpora
    from gensim.models import LdaModel, CoherenceModel, Phrases
    from gensim.models.phrases import Phraser
    from wordcloud import WordCloud
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    import matplotlib.pyplot as plt
    import string
    import pickle
    import os
    import logging

    # Logging setup
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Download necessary NLTK resources
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')

    # Replace these with your own credentials
    reddit = praw.Reddit(client_id='HB9qSD1typ7tlkF78M7dPA',
                         client_secret='rBrpqwT2myyeNBMsUlmuZnXziNoROg',
                         user_agent='MyRedditBot/0.1 by abhinav')

    def fetch_reddit_posts(subreddit_name, limit=100):
        try:
            subreddit = reddit.subreddit(subreddit_name)
            posts = subreddit.hot(limit=limit)
            return [post.title + " " + post.selftext for post in posts]
        except Exception as e:
            logging.error(f"Error fetching posts from r/{subreddit_name}: {e}")
            return []

    def preprocess_text(text):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum()]
        words = [word for word in words if word not in stop_words]
        return [lemmatizer.lemmatize(word) for word in words]

    def visualize_topics(lda_model, num_topics):
        for i in range(num_topics):
            words = dict(lda_model.show_topic(i, topn=10))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
            plt.figure()
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Topic {i}")
        plt.show()

    def save_lda_model(lda_model, dictionary, corpus, model_dir="lda_model"):
        os.makedirs(model_dir, exist_ok=True)
        lda_model.save(os.path.join(model_dir, "model.lda"))
        dictionary.save(os.path.join(model_dir, "dictionary.dict"))
        with open(os.path.join(model_dir, "corpus.pkl"), "wb") as f:
            pickle.dump(corpus, f)
        logging.info(f"LDA model and associated files saved to {model_dir}")

    def load_lda_model(model_dir="lda_model"):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} not found.")
        lda_model = LdaModel.load(os.path.join(model_dir, "model.lda"))
        dictionary = corpora.Dictionary.load(os.path.join(model_dir, "dictionary.dict"))
        with open(os.path.join(model_dir, "corpus.pkl"), "rb") as f:
            corpus = pickle.load(f)
        logging.info(f"LDA model and associated files loaded from {model_dir}")
        return lda_model, dictionary, corpus

    def interactive_visualization(lda_model, corpus, dictionary):
        vis = gensimvis.prepare(lda_model, corpus, dictionary)
        pyLDAvis.show(vis)

    model_dir = "lda_model"
    subreddit_name = 'History'

    if os.path.exists(model_dir):
        print(f"Loading existing model from {model_dir}.")
        lda_model, dictionary, corpus = load_lda_model(model_dir)
    else:
        print(f"No existing model found. Fetching posts from r/{subreddit_name}.")
        posts = fetch_reddit_posts(subreddit_name)

        if not posts:
            print(f"No posts fetched from r/{subreddit_name}. Exiting.")
            return

        print(f"Fetched {len(posts)} posts from r/{subreddit_name}.")

        processed_posts = [preprocess_text(post) for post in posts]

        bigram = Phrases(processed_posts, min_count=5, threshold=100)
        bigram_phraser = Phraser(bigram)
        processed_posts = [bigram_phraser[post] for post in processed_posts]

        dictionary = corpora.Dictionary(processed_posts)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus = [dictionary.doc2bow(post) for post in processed_posts]

        num_topics = 7  # Adjust based on requirements
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

        save_lda_model(lda_model, dictionary, corpus)

    # Print the topics
    topics = lda_model.print_topics(num_words=10)
    for idx, topic in topics:
        print(f"Topic {idx}: {topic}")

    # Compute coherence score
    coherence_model = CoherenceModel(model=lda_model, texts=processed_posts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"Coherence Score: {coherence_score}")

    # Visualize topics using WordClouds
    visualize_topics(lda_model, 7)

    # Interactive visualization
    interactive_visualization(lda_model, corpus, dictionary)

if __name__ == "__main__":
    run()
