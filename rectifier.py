import json
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Status lines use ✓/✗; make sure a non-UTF-8 console (e.g. Windows cp1252) can't crash the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rectification_system import rectify, split_annotations, finalize
from llm_client import get_client

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
MAX_WORKERS = int(os.getenv("RECTIFIER_WORKERS", "6"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def load_mapping():
    with open(BASE_DIR / 'article_mapping.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def get_article_mapping(article_id: str):
    # Find the article by ID
    article_data = next((a for a in load_mapping() if a['article_id'] == article_id), None)
    if not article_data:
        raise ValueError(f"Article {article_id} not found in mapping")

    return article_data


def _read(rel_path: str) -> str:
    with open(BASE_DIR / rel_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_ai_generated_article(article_id: str):
    return _read(get_article_mapping(article_id)['ai_generated_file'])


def get_source_article(article_id: str):
    return _read(get_article_mapping(article_id)['source_file'])


def save_rectified_article(article_id: str, rectified_content: str):
    mapping = get_article_mapping(article_id)
    output_path = BASE_DIR / mapping['rectified_file']

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rectified_content)


def _save_trace(article_id: str, trace: dict):
    try:
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_DIR / f"{article_id}.json", 'w', encoding='utf-8') as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def rectify_article(article_id: str):
    """
    Rectify an AI-generated article.

    Args:
        article_id: ID of the article (e.g., 'article_001')

    Returns:
        str: The rectified article content
    """

    ai_generated_content = get_ai_generated_article(article_id)
    try:
        source_content = get_source_article(article_id)
    except Exception:
        source_content = ""

    # PLUG YOUR CUSTOM RECTIFIER HERE
    rectified_content, trace = rectify(ai_generated_content, source_content, article_id)
    ###################################

    save_rectified_article(article_id, rectified_content)
    _save_trace(article_id, trace)

    n_edits = sum(len(p["applied"]) for p in trace.get("passes", []))
    status = "✗ (fallback: annotations stripped only)" if trace.get("error") else "✓"
    print(f"{status} Rectified {article_id} ({n_edits} edits)", flush=True)
    return rectified_content


def _safe_rectify(article_id: str):
    try:
        rectify_article(article_id)
        return True
    except Exception as e:
        print(f"✗ Error processing {article_id}: {e}", flush=True)
        # Never leave an article without output: fall back to the stripped AI text.
        try:
            body, _ = split_annotations(get_ai_generated_article(article_id))
            save_rectified_article(article_id, finalize(body))
        except Exception as e2:
            print(f"✗ Could not write fallback for {article_id}: {e2}", flush=True)
        return False


def process(articles):
    start = time.time()
    total = len(articles)
    ok = 0
    print(f"Processing {total} articles with {MAX_WORKERS} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_safe_rectify, a['article_id']) for a in articles]
        for fut in as_completed(futures):
            ok += bool(fut.result())

    try:
        usage = get_client().usage
    except Exception:
        usage = {}
    missing = [a['article_id'] for a in articles if not (BASE_DIR / a['rectified_file']).exists()]
    print(f"\n{'='*50}")
    print(f"Completed! Processed {total} articles ({ok} ok, {total - ok} with errors) "
          f"in {time.time() - start:.0f}s.")
    if usage:
        print(f"LLM usage: {usage.get('calls', 0)} calls, {usage.get('prompt_tokens', 0)} prompt + "
              f"{usage.get('completion_tokens', 0)} completion tokens")
    if missing:
        print(f"Missing outputs: {missing}")
    print(f"{'='*50}")


def test_rectifier(count: int):
    """
    Test the rectification system on a subset of articles.

    Args:
        count: Number of articles to test (default: 16)
    """
    process(load_mapping()[:count])


def rectify_all():
    """
    Generate rectified articles for all articles in the mapping (104).
    """
    process(load_mapping())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rectify AI-generated articles by fixing errors and inaccuracies."
    )
    parser.add_argument(
        'command',
        choices=['test', 'rectify-all'],
        help='Command to execute: "test" to process the first N articles, "rectify-all" to process all articles'
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
        rectify_all()
