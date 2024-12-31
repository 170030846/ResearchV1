def run():
    import praw
    import networkx as nx
    import matplotlib.pyplot as plt
    from networkx.algorithms.community import louvain_communities

    # Replace these with your own credentials
    reddit = praw.Reddit(client_id='HB9qSD1typ7tlkF78M7dPA',
                        client_secret='rBrpqwT2myyeNBMsUlmuZnXziNoROg',
                        user_agent='MyRedditBot/0.1 by abhinav')

    def fetch_reddit_posts(subreddit_name, limit=10):
        subreddit = reddit.subreddit(subreddit_name)
        posts = subreddit.hot(limit=limit)
        post_data = []
        for post in posts:
            post_data.append({
                'id': post.id,
                'title': post.title,
                'author': post.author.name if post.author else 'N/A',
                'num_comments': post.num_comments,
                'url': post.url
            })
        return post_data

    def build_network_graph(posts):
        G = nx.Graph()
        for post in posts:
            submission = reddit.submission(id=post['id'])
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list():
                author = comment.author.name if comment.author else 'N/A'
                G.add_node(author)
                
                if comment.parent_id != comment.link_id:  # It's a reply
                    parent_comment = reddit.comment(id=comment.parent_id.split('_')[1])
                    parent_author = parent_comment.author.name if parent_comment.author else 'N/A'
                    G.add_node(parent_author)
                    G.add_edge(author, parent_author)
        
        return G

    def analyze_network(G):
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        centrality_measures = {
            'degree': degree_centrality,
            'betweenness': betweenness_centrality,
            'closeness': closeness_centrality
        }
        
        top_influencers = {}
        for measure, centrality in centrality_measures.items():
            sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            top_influencers[measure] = sorted_centrality
        
        return top_influencers

    def community_detection(G):
        communities = louvain_communities(G)
        return communities

    def visualize_network(G):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        node_sizes = [1000 * G.degree(node) for node in G]  # Adjust node size by degree

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_labels(G, pos, font_size=8)
        plt.title("Reddit Interaction Network")
        plt.show()

    subreddit_name = 'History'
    posts = fetch_reddit_posts(subreddit_name)
    print(f"Fetched {len(posts)} posts from r/{subreddit_name}.")

    G = build_network_graph(posts)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    top_influencers = analyze_network(G)

    for measure, influencers in top_influencers.items():
        print(f"\nTop Influential Users by {measure.capitalize()} Centrality:")
        for user, score in influencers:
            print(f"{user}: {score:.4f}")

    # Community detection
    communities = community_detection(G)
    print("\nDetected Communities:")
    for idx, community in enumerate(communities):
        print(f"Community {idx + 1}: {', '.join(community)}")

    # Visualize the network
    visualize_network(G)

    # Plot Degree Distribution
    degrees = [G.degree(node) for node in G]
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1))
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    run()
