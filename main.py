import argparse
from analysis import sentiment_analysis, engagement_metrics, network_analysis, content_analysis

def main():
    parser = argparse.ArgumentParser(description="Run various analyses on Reddit data.")
    parser.add_argument('analysis', choices=['sentiment', 'engagement', 'network', 'content'], help="Choose the type of analysis to run")
    args = parser.parse_args()

    if args.analysis == 'sentiment':
        sentiment_analysis.run()
    elif args.analysis == 'engagement':
        engagement_metrics.run()
    elif args.analysis == 'network':
        network_analysis.run()
    elif args.analysis == 'content':
        content_analysis.run()

if __name__ == "__main__":
    main()