import json
import argparse
from pathlib import Path
from rectification_system import run

def get_article_mapping(article_id: str):
    # Load article mapping to get file paths
    with open('article_mapping.json', 'r') as f:
        articles = json.load(f)
    
    # Find the article by ID
    article_data = next((a for a in articles if a['article_id'] == article_id), None)
    if not article_data:
        raise ValueError(f"Article {article_id} not found in mapping")
    
    return article_data

def get_ai_generated_article(article_id: str):
    # Read the AI-generated article
    _mapping = get_article_mapping(article_id)
    fpath = _mapping['ai_generated_file']
    with open(fpath, 'r', encoding='utf-8') as f:
        article = f.read()
    return article

def save_rectified_article(article_id: str, rectified_content: str):
    mapping = get_article_mapping(article_id)
    fpath = mapping['rectified_file']
    
    # Ensure output directory exists
    output_path = Path(fpath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(rectified_content)

def rectify_article(article_id: str):
    """
    Rectify an AI-generated article.
    
    Args:
        article_id: ID of the article (e.g., 'article_001')
    
    Returns:
        str: The rectified article content
    """
    
    ai_generated_content = get_ai_generated_article(article_id)
    
    # PLUG YOUR CUSTOM RECTIFIER HERE
    rectified_content = run(ai_generated_content)
    ###################################
    
    save_rectified_article(article_id, rectified_content)
    
    print(f"✓ Rectified {article_id}")
    return rectified_content


def test_rectifier(count: int):
    """
    Test the rectification system on a subset of articles.
    
    Args:
        count: Number of articles to test (default: 16)
    """
    # Load article mapping
    with open('article_mapping.json', 'r') as f:
        articles = json.load(f)
    
    # Test on first 'count' articles
    for i, article in enumerate(articles[:count]):
        if i >= count:
            break
        
        article_id = article['article_id']
        
        print(f"\nProcessing {article_id} ({i+1}/{count})...")
        
        try:
            rectify_article(article_id)
        except Exception as e:
            print(f"✗ Error processing {article_id}: {str(e)}")


def rectify_all():
    """
    Generate rectified articles for all 100 articles.
    """
    # Load article mapping
    with open('article_mapping.json', 'r') as f:
        articles = json.load(f)
    
    total = len(articles)
    
    for i, article in enumerate(articles):
        article_id = article['article_id']
        
        print(f"\nProcessing {article_id} ({i+1}/{total})...")
        
        try:
            rectify_article(article_id)
        except Exception as e:
            print(f"✗ Error processing {article_id}: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"Completed! Processed {total} articles.")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rectify AI-generated articles by fixing errors and inaccuracies."
    )
    parser.add_argument(
        'command',
        choices=['test', 'rectify-all'],
        help='Command to execute: "test" to process first 16 articles, "rectify-all" to process all 100 articles'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=16,
        help='Number of articles to test (only applicable for "test" command, default: 16)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'test':
        print(f"Testing rectification system on first {args.count} articles...")
        test_rectifier(count=args.count)
    elif args.command == 'rectify-all':
        print("Processing all 100 articles...")
        rectify_all()