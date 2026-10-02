"""
Article Rectification System — "surgical" editor.

The LLM never rewrites the article. It only returns a list of minimal
find -> replace edits (JSON), which are applied deterministically in code, so
every character the model did not explicitly target is preserved verbatim.

Pipeline per article:
  1. Split the AI article into body + trailing "Error Annotations" block
     (generator artefact; it must never reach the output, but it is a useful,
     unverified hint about which spans were corrupted).
  2. Edit pass: source + body + hints -> JSON edits.
  3. Apply edits with exact matching, falling back to whitespace/quote/dash
     tolerant matching. Edits that still cannot be located are sent back to the
     model once to re-quote the span.
  4. Review pass: the corrected body is re-checked against the source for any
     remaining contradictions (catches misses, esp. when there are no hints).
  5. Clean-up (spacing left by deletions, trailing whitespace) and return.

Any failure degrades gracefully to the best text produced so far (at minimum
the body with the annotation block stripped), so a file is always written.
"""

import json
import logging
import os
import re

from llm_client import LLMError, get_client

log = logging.getLogger("rectifier")

_temp = os.getenv("LLM_TEMPERATURE", "").strip()
TEMPERATURE = float(_temp) if _temp else None  # None = provider default
EDIT_EFFORT = os.getenv("LLM_REASONING_EFFORT", "medium")  # reasoning effort of the main edit pass

ANNOTATION_RE = re.compile(r"(?m)^[ \t#*]*Error Annotations\s*:?\s*\**\s*:?")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def split_annotations(text: str):
    """Return (body, raw_annotation_text). raw is '' when there is no block."""
    text = text.replace("\r\n", "\n")
    m = ANNOTATION_RE.search(text)
    if not m:
        return text.rstrip(), ""
    body = text[:m.start()].rstrip()
    raw = text[m.end():].strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
    return body, raw


def parse_annotations(raw: str):
    """Best-effort parse of the annotation JSON into a list of dicts."""
    if not raw:
        return []
    candidates = [raw, re.sub(r'\\"', '"', raw)]
    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass
    # Fallback: pull out "error" / "correction" fields with a regex.
    items = []
    for m in re.finditer(r'"error"\s*:\s*\\?"(.*?)\\?",\s*\n\s*"correction"\s*:\s*\\?"(.*?)\\?",', raw, re.S):
        items.append({"error": m.group(1), "correction": m.group(2)})
    return items


def format_hints(annotations, raw):
    if annotations:
        lines = []
        for i, a in enumerate(annotations, 1):
            lines.append(
                f'{i}. flagged span: {a.get("error", "")!s}\n'
                f'   suggested fix (may be paraphrased or imprecise): {a.get("correction", "")!s}'
            )
        return "\n".join(lines)
    return raw.strip()


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a surgical fact-correction editor. An AI agent wrote an ARTICLE based on a SOURCE document, and afterwards a small number of errors were injected into it: wrong numbers, dates, years, days, names, places, organisations, platforms, titles, currencies, swapped attributes, flipped negations or sentiment, wrong attributions, exaggerated scale, or an invented clause bolted onto an otherwise true sentence.

Your job is to undo exactly those injected errors and nothing else. You do NOT rewrite. You output a list of minimal edits that a program applies with exact string replacement.

What counts as an error:
- A statement the SOURCE contradicts (different number, name, date, place, entity, outcome, negation, sentiment).
- A span flagged in the HINTS (when hints are provided), after checking it against the SOURCE.
What is NOT an error (leave it alone):
- Details, sentences, sections or headings that the SOURCE simply does not mention. The ARTICLE legitimately contains background knowledge beyond the SOURCE. Never delete or change something just because the SOURCE lacks it.
- Style, grammar, wording, formatting, rounding, vague-but-compatible phrasing.

How to fix:
1. Change only the erroneous words; everything else stays character-identical (wording, punctuation, quote characters, number style, spelling, headings).
2. Swap the wrong value for the correct one, but keep the ARTICLE's own notation and spelling, never the SOURCE's. Change only the wrong characters:
   "$1,34,900" -> "₹1,34,900" (not "Rs. 1,34,900");  "30,000-nits" -> "3,000-nits" (not "3,000 nits");  "1,000cc" -> "1,200cc" (not "1,200 cc").
   Prefer the specific value (e.g. "Instagram") over a vaguer phrase (e.g. "social media").
3. A corrupted sentence is corrected in place (fix its facts, keep its structure). Do not delete a sentence that just needs its facts fixed.
4. Delete text only when it is an injected add-on that has no true counterpart. Typical shapes:
   - an appended clause: "..., with some critics even calling it a light-hearted comedy." -> delete the clause;
   - an invented contrast: "X, rather than Y," / "instead of Y" / "— not Y —" where the contrast itself is false -> keep only the true part;
   - an invented event, venue or date tacked onto a true statement: "unveiled its phones at WWDC in June 2025, the ..." -> "unveiled its phones, the ..." (delete the invented phrase rather than substituting a different event).
   Delete just that clause (with its adjoining comma/space) and nothing more.
5. Never add new information that was not in the erroneous span, and never copy explanatory or hedging language into the article (e.g. "(not X)", "— not a ...", "also referred to as ... in some references", "purportedly", "no claim of ...", "did not deny ... in that regard"). The article must read as if the error had never been there.
6. When you change a noun, keep the sentence grammatical with the smallest change (e.g. "an above-par" -> "a below-par").

Output format — a single JSON object:
{"edits": [{"find": "<exact verbatim substring of the ARTICLE>", "replace": "<replacement text>", "hint": <number of the hint this edit fixes, or null if unflagged>, "reason": "<short justification citing the source/hint>"}]}
- "find" must be copied character-for-character from the ARTICLE (same quotes, dashes, spacing). Keep it short (a few words around the error) but unambiguous; it must not span a line break.
- If the same wrong fact appears several times, give one edit per occurrence (every occurrence of "find" is replaced, so a shared find fixes all of them).
- Edits must not overlap.
- Return {"edits": []} if nothing needs fixing."""


def build_edit_prompt(source: str, body: str, hints: str) -> str:
    hint_block = (
        "HINTS — the generator's own log of the errors it injected. The \"flagged span\" is the corrupted "
        "text; the \"suggested fix\" usually contains the original value and wording, so when it is consistent "
        "with the SOURCE reuse its words for the corrected fact. Suggested fixes often also contain the "
        "annotator's commentary — parentheticals, \"— not X\", \"also referred to as ... in some references\", "
        "\"purportedly\", \"no claim of X\", \"described as Y\", \"did not ... in that regard\". Never copy that "
        "commentary: apply the minimal fix instead (correct the facts in place, or delete the injected add-on "
        "clause). Fix every hint, and tag each edit with "
        "the number of the hint it fixes (the same wrong fact repeated elsewhere in the article belongs to "
        "the same hint). The hints are the complete list of injected errors: do NOT edit anything that no "
        "hint covers.\n"
        f"<hints>\n{hints}\n</hints>\n\n"
        if hints else
        "No hints are available. Carefully compare every factual claim in the ARTICLE (numbers, dates, "
        "names, places, organisations, currencies, materials, attributions, negations, comparisons, "
        "superlatives, added events/locations) against the SOURCE, and fix only clear contradictions "
        "or clearly injected add-on claims.\n\n"
    )
    return (
        f"<source>\n{source.strip()}\n</source>\n\n"
        f"<article>\n{body}\n</article>\n\n"
        f"{hint_block}"
        "Return the JSON object of minimal edits."
    )


REVIEW_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You are now doing a final review of an article that has ALREADY been corrected once. Report only remaining, clear-cut factual errors where the SOURCE explicitly states something different. Do not re-edit text consistent with the SOURCE, do not delete sentences or headings, and do not touch content the SOURCE does not cover. An empty list is the expected answer for most articles."""


def build_review_prompt(source: str, body: str) -> str:
    return (
        f"<source>\n{source.strip()}\n</source>\n\n"
        f"<article>\n{body}\n</article>\n\n"
        "Return the JSON object of any remaining minimal edits (usually none)."
    )


# --------------------------------------------------------------------------- #
# LLM calls
# --------------------------------------------------------------------------- #
def parse_edits(content: str):
    content = (content or "").strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            raise
        data = json.loads(m.group(0))
    edits = data.get("edits", []) if isinstance(data, dict) else data
    out, seen = [], set()
    for e in edits or []:
        if isinstance(e, dict) and isinstance(e.get("find"), str) and isinstance(e.get("replace"), str):
            key = (e["find"], e["replace"])
            if e["find"] and e["find"] != e["replace"] and key not in seen:
                seen.add(key)
                out.append({"find": e["find"], "replace": e["replace"], "hint": e.get("hint"),
                            "reason": str(e.get("reason", ""))})
    return out


def ask_for_edits(system: str, user: str, effort: str):
    client = get_client()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_err = None
    for _ in range(3):
        try:
            content, finish = client.chat(messages, reasoning_effort=effort, temperature=TEMPERATURE)
            return parse_edits(content)
        except (json.JSONDecodeError, ValueError, AttributeError, LLMError) as e:
            last_err = e
    raise RuntimeError(f"Could not obtain valid JSON edits: {last_err}")


# --------------------------------------------------------------------------- #
# Applying edits
# --------------------------------------------------------------------------- #
_QUOTES1, _QUOTES2, _DASHES = "'‘’`´", '"“”„', "-‐‑‒–—―−"
_EQUIV = {
    **{c: "[" + re.escape(_QUOTES1) + "]" for c in _QUOTES1},
    **{c: "[" + re.escape(_QUOTES2) + "]" for c in _QUOTES2},
    **{c: "[" + re.escape(_DASHES) + "]" for c in _DASHES},
}


def _tolerant_pattern(find: str) -> str:
    parts = []
    for ch in find.strip():
        if ch.isspace():
            ws = r"[ \t ]+"  # never match across a line break
            if not parts or parts[-1] != ws:
                parts.append(ws)
        else:
            parts.append(_EQUIV.get(ch, re.escape(ch)))
    return "".join(parts)


def _locate(text: str, find: str):
    """Return list of (start, end) spans for `find` in `text`."""
    spans, i = [], text.find(find)
    while i != -1:
        spans.append((i, i + len(find)))
        i = text.find(find, i + len(find))
    if spans:
        return spans
    for flags in (0, re.IGNORECASE):
        spans = [(m.start(), m.end()) for m in re.finditer(_tolerant_pattern(find), text, flags)]
        if spans:
            return spans
    return []


def _tidy_deletion(text: str, pos: int) -> str:
    """Clean whitespace/punctuation artefacts around a deletion point."""
    lo, hi = max(0, pos - 3), min(len(text), pos + 3)
    window = text[lo:hi]
    fixed = re.sub(r"[ \t]{2,}", " ", window)
    fixed = re.sub(r" ([,.;:!?])", r"\1", fixed)
    fixed = re.sub(r",([.;:!?])", r"\1", fixed)
    return text[:lo] + fixed + text[hi:]


_DUP_WORD = re.compile(r"\b(\w+)(\s+)\1\b", re.IGNORECASE)


def _fix_seam(before: str, after: str, start: int, end: int) -> str:
    """Collapse a doubled word (e.g. "the the") that an edit created at its seams."""
    lo, hi = max(0, start - 30), min(len(after), end + 30)
    window = after[lo:hi]
    old_window = before[max(0, start - 30):start + (end - start) + 30]
    for m in _DUP_WORD.finditer(window):
        if m.group(0).lower() not in old_window.lower():
            return after[:lo + m.start()] + m.group(1) + after[lo + m.end():]
    return after


def _fix_indefinite_article(text: str, start: int) -> str:
    """Make an "a"/"an" right before an edited span agree with its new first word."""
    m = re.search(r"\b([Aa]n?)([ \t]+)$", text[max(0, start - 6):start])
    nxt = re.match(r"[A-Za-z]+", text[start:])
    if not m or not nxt:
        return text
    word = nxt.group(0).lower()
    if word.startswith(("uni", "use", "usu", "eu", "one", "once", "h")):
        return text  # ambiguous pronunciation; leave as is
    want = "an" if word[0] in "aeiou" else "a"
    have = m.group(1)
    if have.lower() == want:
        return text
    new = want.capitalize() if have[0].isupper() else want
    art_start = start - len(m.group(0))
    return text[:art_start] + new + m.group(2) + text[start:]


_UNIT_RE = re.compile(r"(\d)([ \-]?)([A-Za-z]{1,6})\b")


def _preserve_format(find: str, repl: str, text: str = "") -> str:
    """Undo notation drift the model copies from the source (the article's style wins)."""
    # Number-unit joining: "1,000cc" -> "1,200 cc" becomes "1,200cc"; "30,000-nits" keeps its hyphen.
    seps = {m.group(3).lower(): m.group(2) for m in _UNIT_RE.finditer(find)}
    if seps:
        def join(m):
            sep = seps.get(m.group(3).lower())
            return m.group(1) + (sep if sep is not None else m.group(2)) + m.group(3)
        repl = _UNIT_RE.sub(join, repl)
    # Currency notation: keep "₹" if the article used it.
    if "₹" in (find + text) and "₹" not in repl:
        repl = re.sub(r"\b(?:Rs\.?|INR)\s?(?=\d)", "₹", repl)
    # Keep a sentence-final full stop the model dropped from a non-empty replacement.
    if find.rstrip().endswith(".") and repl.strip() and not re.search(r"[.!?…\"'”’)]\s*$", repl):
        repl = repl.rstrip() + "."
    return repl


def apply_edits(text: str, edits, max_find=400, max_deletion=250):
    """Apply edits; returns (text, applied, failed, rejected).

    failed   = could not be located (worth a repair round)
    rejected = located but blocked by safety guards (too large / multi-line)
    """
    applied, failed, rejected = [], [], []
    for e in edits:
        find = e["find"]
        repl = _preserve_format(find, e["replace"], text)
        deletion = not repl.strip()
        if "\n" in find.strip() or "\n" in repl.strip():
            rejected.append({**e, "error": "spans a line break"})
            continue
        if len(find) > max_find or (deletion and len(find) > max_deletion):
            rejected.append({**e, "error": "edit too large"})
            continue
        spans = _locate(text, find)
        if not spans:
            failed.append({**e, "error": "not found"})
            continue
        for start, end in reversed(spans):
            original = text[start:end]
            before = text
            text = text[:start] + repl + text[end:]
            if deletion:
                text = _tidy_deletion(text, start)
            else:
                text = _fix_seam(before, text, start, start + len(repl))
                if repl[:1].isalpha() and repl[:1].lower() != original[:1].lower():
                    text = _fix_indefinite_article(text, start)
            applied.append({**e, "matched": original})
    return text, applied, failed, rejected


def repair_failed(source: str, text: str, failed):
    """Ask the model once to re-quote edits whose `find` was not located."""
    listing = json.dumps([{k: f.get(k) for k in ("find", "replace", "hint", "reason")} for f in failed],
                         ensure_ascii=False, indent=1)
    user = (
        f"<source>\n{source.strip()}\n</source>\n\n<article>\n{text}\n</article>\n\n"
        "The following edits could not be applied because their \"find\" text is not an exact "
        "substring of the ARTICLE (or the fact was already corrected). For each one that is still "
        "needed, return a corrected edit whose \"find\" is copied exactly from the ARTICLE. "
        "Keep each edit's \"hint\" number. Drop edits that are no longer needed.\n"
        f"<failed_edits>\n{listing}\n</failed_edits>"
    )
    return ask_for_edits(SYSTEM_PROMPT, user, "low")


def unresolved_hints(annotations, original: str, current: str):
    """Hints whose flagged span occurs verbatim in the original article and still survives."""
    out = []
    for i, a in enumerate(annotations, 1):
        err = str(a.get("error", "")).strip().strip("\"'")
        if len(err) >= 3 and err in original and err in current and err not in str(a.get("correction", "")):
            out.append((i, a))
    return out


def repair_unresolved(source: str, text: str, pending):
    """One targeted call for flagged spans that are still present after the edit pass."""
    listing = "\n".join(
        f'{i}. flagged span: {a.get("error", "")}\n   suggested fix (may be paraphrased or imprecise): '
        f'{a.get("correction", "")}' for i, a in pending)
    user = (
        f"<source>\n{source.strip()}\n</source>\n\n<article>\n{text}\n</article>\n\n"
        "These flagged errors are still present verbatim in the ARTICLE (the same wrong fact may have been "
        "fixed elsewhere, e.g. in a heading, but not here). Return the minimal edits that fix each remaining "
        "occurrence, tagged with its hint number. Follow all the editing rules (no commentary, keep the "
        "article's notation, delete only injected add-on clauses).\n"
        f"<hints>\n{listing}\n</hints>"
    )
    return ask_for_edits(SYSTEM_PROMPT, user, "low")


def finalize(text: str) -> str:
    # Only outer whitespace is normalised; inner text is left byte-for-byte as edited.
    return text.strip()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _gate_unflagged(edits, has_hints):
    """With hints available, unflagged edits were measured to be mostly false positives: drop them."""
    if not has_hints:
        return edits, []
    keep, dropped = [], []
    for e in edits:
        (keep if e.get("hint") not in (None, "", "null", 0) else dropped).append(e)
    return keep, [{**e, "error": "not covered by any hint"} for e in dropped]


def rectify(ai_generated_content: str, source_content: str, article_id: str = "", review: str = "auto"):
    """
    Rectify an AI-generated article against its source.

    review: "auto" (second pass only when the article has no hints), "always" or "never".

    Returns (rectified_text, trace). Never raises: on failure it returns the best
    text produced so far (at least the article with its annotation block removed).
    """
    body, raw_ann = split_annotations(ai_generated_content)
    current = body
    trace = {"article_id": article_id, "passes": []}
    if not source_content.strip():
        log.warning("%s: no source text, only stripping annotations", article_id)
        trace["error"] = "no source"
        return finalize(current), trace

    try:
        annotations = parse_annotations(raw_ann)
        hints = format_hints(annotations, raw_ann)
        if hints.strip() in ("", "[]"):
            hints = ""
        trace["hints"] = bool(hints)
        edits = ask_for_edits(SYSTEM_PROMPT, build_edit_prompt(source_content, body, hints), EDIT_EFFORT)
        edits, rejected = _gate_unflagged(edits, bool(hints))
        current, applied, failed, rejected_now = apply_edits(current, edits)
        rejected += rejected_now
        if failed:
            retry, gated = _gate_unflagged(repair_failed(source_content, current, failed), bool(hints))
            current, applied2, failed, rejected2 = apply_edits(current, retry)
            applied += applied2
            rejected += gated + rejected2
        trace["passes"].append({"name": "edit", "applied": applied, "failed": failed, "rejected": rejected})

        pending = unresolved_hints(annotations, body, current) if hints else []
        if pending:
            try:
                edits, gated = _gate_unflagged(repair_unresolved(source_content, current, pending), True)
                current, applied, failed, rejected = apply_edits(current, edits)
                trace["passes"].append({"name": "coverage", "pending": [i for i, _ in pending],
                                        "applied": applied, "failed": failed, "rejected": gated + rejected})
            except Exception as e:
                log.warning("%s: coverage repair failed: %s", article_id, e)

        if review == "always" or (review == "auto" and not hints):
            try:
                edits = ask_for_edits(REVIEW_SYSTEM_PROMPT, build_review_prompt(source_content, current), "medium")
                current, applied, failed, rejected = apply_edits(current, edits, max_find=200, max_deletion=80)
                trace["passes"].append({"name": "review", "applied": applied, "failed": failed, "rejected": rejected})
            except Exception as e:  # review is best-effort
                log.warning("%s: review pass failed: %s", article_id, e)
                trace["review_error"] = str(e)
    except Exception as e:
        log.error("%s: rectification failed, falling back: %s", article_id, e)
        trace["error"] = str(e)

    return finalize(current), trace


def run(ai_generated_content: str, source_content: str = "", article_id: str = "") -> str:
    """Compatibility wrapper matching the original template signature."""
    return rectify(ai_generated_content, source_content, article_id)[0]
