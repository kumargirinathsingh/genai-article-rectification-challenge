"""
Offline evaluation against the human-rectified reference examples.

The 10 references are kept in eval/references/ (rectify-all overwrites
rectified_articles/). For each article we diff, at word level:
  A = AI article body (annotations stripped)   R = human reference   O = our output

  gold edits   = changed blocks of A -> R
  our edits    = changed blocks of A -> O
  recall       = gold edits whose region in R is reproduced exactly in O
  precision    = our edits that overlap some gold edit (the rest are over-edits)
  gap closed   = 1 - dist(O, R) / dist(A, R)   (word-level edit distance)

Usage:
  python evaluate.py               # score existing files in rectified_articles/
  python evaluate.py --run         # rectify the reference articles first, then score
  python evaluate.py --show        # also print the remaining diffs
"""

import argparse
import difflib
import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rectification_system import split_annotations

BASE = Path(__file__).resolve().parent
REF_DIR = BASE / "eval" / "references"


def words(text):
    return re.findall(r"\S+", text)


def blocks(a, b):
    return [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]


def dist(a, b):
    return sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in blocks(a, b))


def overlaps(r1, r2):
    (a1, a2), (b1, b2) = r1, r2
    if a1 == a2 or b1 == b2:  # insertion point: touching counts
        return a1 <= b2 and b1 <= a2
    return a1 < b2 and b1 < a2


def score(article_id, show=False):
    A = words(split_annotations((BASE / "ai_generated_articles" / f"{article_id}.txt").read_text(encoding="utf-8"))[0])
    R = words((REF_DIR / f"{article_id}.txt").read_text(encoding="utf-8"))
    O = words((BASE / "rectified_articles" / f"{article_id}.txt").read_text(encoding="utf-8"))

    gold = blocks(A, R)
    ours = blocks(A, O)
    residual = blocks(O, R)  # ranges in R that we still get wrong: (j1, j2)

    fixed = sum(1 for _, i1, i2, j1, j2 in gold
                if not any(overlaps((j1, j2), (rj1, rj2)) for _, _, _, rj1, rj2 in residual))
    correct_ours = sum(1 for _, i1, i2, _, _ in ours
                       if any(overlaps((i1, i2), (g1, g2)) for _, g1, g2, _, _ in gold))
    base_d, our_d = dist(A, R), dist(O, R)

    if show and residual:
        print(f"--- {article_id} remaining differences (ours -> reference)")
        for tag, i1, i2, j1, j2 in residual:
            print(f"  [{tag}] ours: {' '.join(O[max(0,i1-3):i2+3])!r}\n          ref:  {' '.join(R[max(0,j1-3):j2+3])!r}")
    return {"id": article_id, "gold": len(gold), "fixed": fixed, "ours": len(ours),
            "ours_ok": correct_ours, "base_d": base_d, "our_d": our_d}


def hint_coverage(show=False):
    """For every article with an annotation block: are the flagged error spans gone from the output?

    Only flagged spans that occur verbatim in the AI article (and are not merely quoted
    inside the suggested correction) are checkable."""
    from rectification_system import parse_annotations
    checkable = remaining = 0
    leftovers = []
    for ai_path in sorted((BASE / "ai_generated_articles").glob("*.txt")):
        out_path = BASE / "rectified_articles" / ai_path.name
        if not out_path.exists():
            continue
        body, raw = split_annotations(ai_path.read_text(encoding="utf-8"))
        out = out_path.read_text(encoding="utf-8")
        for a in parse_annotations(raw):
            err = str(a.get("error", "")).strip().strip('"\'')
            if len(err) < 3 or err not in body or err in str(a.get("correction", "")):
                continue
            checkable += 1
            if err in out:
                remaining += 1
                leftovers.append((ai_path.stem, err))
    if show:
        for aid, err in leftovers:
            print(f"  still present in {aid}: {err[:100]!r}")
    print(f"Flagged error spans removed (all articles): {checkable - remaining}/{checkable} "
          f"= {(checkable - remaining) / max(checkable, 1):.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="rectify the reference articles before scoring")
    ap.add_argument("--show", action="store_true", help="print remaining differences")
    ap.add_argument("--ids", nargs="*", help="restrict to these article ids")
    ap.add_argument("--coverage", action="store_true",
                    help="also check, over all articles, that annotated error spans were removed")
    args = ap.parse_args()

    ids = args.ids or sorted(p.stem for p in REF_DIR.glob("*.txt"))
    if args.run:
        import rectifier
        rectifier.process([rectifier.get_article_mapping(i) for i in ids])

    rows = [score(i, args.show) for i in ids]
    print(f"\n{'article':<13}{'gold':>5}{'fixed':>6}{'ours':>6}{'on-target':>10}{'dist base->ours':>17}")
    for r in rows:
        print(f"{r['id']:<13}{r['gold']:>5}{r['fixed']:>6}{r['ours']:>6}{r['ours_ok']:>10}"
              f"{r['base_d']:>9} -> {r['our_d']:<5}")
    g = sum(r["gold"] for r in rows); f = sum(r["fixed"] for r in rows)
    o = sum(r["ours"] for r in rows); ok = sum(r["ours_ok"] for r in rows)
    bd = sum(r["base_d"] for r in rows); od = sum(r["our_d"] for r in rows)
    print(f"\nRecall (gold edits reproduced exactly): {f}/{g} = {f / max(g, 1):.1%}")
    print(f"Precision (our edits on a gold span):   {ok}/{o} = {ok / max(o, 1):.1%}")
    print(f"Gap closed (word edit distance):        {bd} -> {od} = {1 - od / max(bd, 1):.1%}")
    if args.coverage:
        hint_coverage(args.show)


if __name__ == "__main__":
    main()
