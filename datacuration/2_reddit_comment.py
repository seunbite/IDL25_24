from careerpathway.utils import open_json, save_json
import os
import praw
import time
import fire
from tqdm import tqdm
from prawcore import RequestException, ResponseException
import re
from urllib.parse import urlparse, quote
import sys
import locale
import multiprocessing as mp
from functools import partial
import json
import yaml

with open('Career-Pathway/config/api.yaml', 'r') as f:
    config = yaml.safe_load(f)


def extract_submission_id(post_url):
    """Extract submission ID from Reddit URL"""
    if not post_url:
        return None
    
    # Handle different Reddit URL formats
    if '/comments/' in post_url:
        parts = post_url.split('/comments/')
        if len(parts) > 1:
            submission_id = parts[1].split('/')[0]
            return submission_id
    return None

def fetch_all_comments(post_url=None, post_id=None, reddit=None):
    if post_id:    
        submission_id = post_id
    else:
        submission_id = extract_submission_id(post_url)
    
    if not submission_id:
        print(f"Could not extract submission ID from {post_url}")
        return []
    
    try:
        submission = reddit.submission(id=submission_id)
        # Replace MoreComments objects to get all comments
        submission.comments.replace_more(limit=None)
        comments = submission.comments.list()
        comment_list = []
        for comment in comments:
            comment_list.append({
                'id': comment.id,
                'parent_id': comment.parent_id,  # t1_xxx(댓글) or t3_xxx(포스트)
                'link_id': comment.link_id,      # t3_xxx (포스트)
                'body': comment.body,
                'author': str(comment.author) if comment.author else None,
                'score': comment.score,
                'created_utc': comment.created_utc,
            })
        return comment_list
    except (RequestException, ResponseException) as e:
        print(f"Reddit API error for {post_url}: {e}")
        return []
    except Exception as e:
        print(f"Error fetching comments for {post_url}: {e}")
        return []


def process_single_post(post_data, output_path, reddit):
    """Process a single post and save its comments"""
    try:
        post_url = post_data.get('Post URL')
        post_id = post_data.get('ID')
        
        if not post_url:
            return None
            
        comments = fetch_all_comments(post_url=post_url, post_id=post_id, reddit=reddit)
        
        # Add post info to each comment
        for c in comments:
            c['post_id'] = post_id
            c['post_url'] = post_url
        
        result = {**post_data, 'comments': comments}
        
        # Save to file with lock to prevent race conditions
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Rate limiting
        time.sleep(1)
        
        return len(comments)
        
    except Exception as e:
        print(f"Error processing post {post_data.get('ID', 'unknown')}: {e}")
        return 0

# Reddit API 환경변수 필요
def main(
    reddit_data_path: str = "career_data/reddit",
    output_path: str = "career_data/reddit/all_with_comments.jsonl",
    num_processes: int = 2,
    start: int = 0,
):
    # 환경변수 체크
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', None) if os.getenv('REDDIT_CLIENT_ID', None) else config['REDDIT_CLIENT_ID']
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', None) if os.getenv('REDDIT_CLIENT_SECRET', None) else config['REDDIT_CLIENT_SECRET']
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', None) if os.getenv('REDDIT_USER_AGENT', None) else config['REDDIT_USER_AGENT']
    
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        print("Error: Missing Reddit API environment variables")
        return
    
    print(f"Using {num_processes} processes")
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

    data_paths = [os.path.join(reddit_data_path, f) for f in os.listdir(reddit_data_path) if f.endswith('.jsonl') and 'data3_reddit_data' in f]

    total_data = []
    done = open_json(output_path)
    done = [(item['ID'], item['Title']) for item in done] if done else []

    for data_path in data_paths:
        data = open_json(data_path)
        for item in data:
            if (item['ID'], item['Title']) not in done:
                total_data.append(
                    {
                    **item, 
                    'keyword': os.path.basename(data_path).replace('.jsonl', '').replace('data3_reddit_data_', ''),
                    'comments': []
                    }
                )
                done.append((item['ID'], item['Title']))
    
    print(f"Total posts to process: {len(total_data)}", f"Start from {start}")
    
    # Create partial function with output_path
    process_func = partial(process_single_post, output_path=output_path, reddit=reddit)
    
    # Use multiprocessing with progress bar
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, total_data[start:]),
            total=len(total_data[start:]),
            desc="Processing posts"
        ))
    
    total_comments = sum([r for r in results if r is not None])
    print(f"Processing complete! Total comments collected: {total_comments}")


if __name__ == "__main__":
    fire.Fire(main)