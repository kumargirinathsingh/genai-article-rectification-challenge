# Write-up: AI Article Rectification

## Approach in one line

The LLM never rewrites the article. It only proposes minimal `find → replace` edits as JSON, and deterministic code applies them, so every character the model did not explicitly target stays byte-for-byte identical.

## Pipeline

1. **Parse.** I strip the trailing `Error Annotations` block that 101 of the 104 AI articles carry. It must not reach the output, but I parse it leniently, since some blocks are malformed or empty, and use it as a hint list of corrupted spans.
2. **Edit pass.** `gpt-oss-120b` (reasoning effort medium) receives the source, the article and the hints. It returns edits, each tagged with the hint it fixes.
3. **Apply in code.**
   - Exact matching first, then a fallback that tolerates quote, dash and whitespace variants but never crosses a line break.
   - Safety guards reject multi-line or very large edits.
   - Seam clean-up: no doubled words, no stray spaces or commas after deletions, a/an agreement.
   - The article's own notation is restored when the model copies the source's (₹ vs "Rs.", "1,200cc" vs "1,200 cc").
4. **Targeted repairs, only when needed.**
   - An edit whose text cannot be located is re-quoted once.
   - A flagged error span still present afterwards gets one follow-up call. Typically the same error appeared in both a headline and the body.
5. **Review pass, only for articles without hints.** A conservative second check against the source.
6. **Never fail.** Any error falls back to the best text so far, at minimum the article with the annotations removed. Articles run in parallel with retries and backoff, so `rectify-all` always writes all 104 files.

## Key assumptions

- **The references are the original clean articles, with errors injected afterwards.** In the diffs, fabricated add-ons are deleted, not rewritten, and corrupted facts are fixed in place.
- **Annotations are reliable locators but imperfect corrections.** Their suggested fixes sometimes contain commentary ("purportedly", "(not X)"). So the source supplies the fact, and the prompt forbids copying commentary into the article.
- **"Not in the source" is not an error.** The articles legitimately include background knowledge. Only facts the source contradicts, or flagged spans, are edited.

## Design choices backed by evaluation

I built the metric first (`evaluate.py`). It diffs the 10 human references at word level and also checks that flagged errors were removed across all 101 annotated articles. Each change below was kept or dropped based on those numbers.

| Change | Effect |
| --- | --- |
| Conservative prompt and size guards | Fixed an early version that deleted whole correct sections. Precision went from 73% to 96%. |
| Hint gating: drop untagged edits when hints exist | Every unflagged edit in evaluation was a false positive. |
| Review pass only when there are no hints | With hints, the review pass caused over-edits. |
| Lower temperature, `reasoning_effort=high` | No improvement, higher cost. Rejected. |

**Final results:**
- About 96% edit precision.
- About 60% of human edits reproduced word for word. Most of the gap is sentence rewording that can't be inferred from the source.
- 100% (367/367) of flagged errors removed.
- 3 articles identical to the human version.
- A full run takes 2–3 minutes and costs about $0.10. Total spend was $0.43 of the $3 budget.

## Engineering notes

- **I replaced the `litellm` SDK with plain `requests`.** The pinned version is no longer installable from PyPI. The client accepts LiteLLM-style model names and resolves them to whatever the endpoint accepts.
- **Console output is UTF-8-safe**, so the status symbols can't crash a Windows console.
- **Per-article traces of every applied, failed and rejected edit go to `logs/`**, for auditability.
