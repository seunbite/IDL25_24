import os
import praw
from praw.models import MoreComments

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')

# Ensure user agent is properly encoded and doesn't contain Unicode characters
if REDDIT_USER_AGENT:
    # Remove any Unicode characters and ensure it's ASCII
    REDDIT_USER_AGENT = REDDIT_USER_AGENT.encode('ascii', 'ignore').decode('ascii')
else:
    REDDIT_USER_AGENT = "MyRedditBot/1.0"

print(f"Client ID: {REDDIT_CLIENT_ID}")
print(f"Client Secret: {'*' * len(REDDIT_CLIENT_SECRET) if REDDIT_CLIENT_SECRET else 'None'}")
print(f"User Agent: {REDDIT_USER_AGENT}")

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    check_for_updates=False,
    check_for_async=False,
    requestor_kwargs={'timeout': 30}
)

url = "https://www.reddit.com/r/funny/comments/3g1jfi/buttons/"

try:
    submission = reddit.submission("3g1jfi")
    
    print(f"Title: {submission.title}")
    print(f"Score: {submission.score}")
    print(f"Number of comments: {submission.num_comments}")
    print("\n" + "="*50 + "\n")
    
    # Replace the comment loop with a more robust version
    submission.comments.replace_more(limit=0)  # Remove MoreComments objects
    
    for i, top_level_comment in enumerate(submission.comments):
        if hasattr(top_level_comment, 'body') and top_level_comment.body:
            print(f"Comment {i+1}:")
            print(f"Author: {top_level_comment.author}")
            print(f"Score: {top_level_comment.score}")
            print(f"Body: {top_level_comment.body}")
            print("-" * 30)
            
except Exception as e:
    print(f"Error accessing Reddit: {e}")
    print("Please check your environment variables and Reddit API credentials.")