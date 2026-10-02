# AI Article Rectification Challenge

## Solution: a "surgical" AI editor

### Quick start

```bash
pip install -r requirements.txt
cp env.example .env              # fill in LLM_API_KEY / LLM_API_BASE / LLM_MODEL_NAME
python rectifier.py rectify-all  # writes all 104 files to rectified_articles/
python evaluate.py --coverage    # optional: score against the human references
```

No third-party services are used beyond the provided LLM endpoint. A full run makes about 110 LLM calls (roughly 450k tokens, about $0.10) and takes 2–3 minutes with 6 parallel workers.

### Core idea: the LLM proposes edits, code applies them

Asking an LLM to "return the corrected article" lets it paraphrase. This pipeline never lets the model write the article. It asks only for a JSON list of minimal edits:

```json
{"edits": [{"find": "63 million", "replace": "165 million", "hint": 2, "reason": "source: 165 million km²"}]}
```

Code then applies each edit by exact string replacement. Every character the model did not explicitly target is preserved byte-for-byte.

### Pipeline (`rectification_system.py`)

1. **Split off the annotation block.** 101 of the 104 AI articles end with an `**Error Annotations:**` JSON block, the generator's own log of the errors it injected. It is removed from the output and parsed leniently (some blocks have broken escaping or are empty) to serve as hints.
2. **Edit pass** (`gpt-oss-120b`, reasoning effort medium). The model receives the source, the article body and the hints, and returns tagged edits. The prompt encodes the rules learned from the 10 references:
   - Correct facts in place.
   - Delete only injected add-on clauses (`", with critics calling it a comedy"`, `"rather than X"`, an invented venue). Never delete content just because the source doesn't mention it.
   - Keep the article's own notation (`₹1,34,900`, not the source's `Rs. 1,34,900`).
   - Never copy the annotator's commentary ("purportedly", "(not X)").
3. **Deterministic application with guards** (`apply_edits`):
   - Exact match first, then a matcher that tolerates quote, dash and whitespace variants but never crosses a line break.
   - Edits that are too large or span multiple lines are rejected.
   - Seam clean-up: no doubled words ("the the"), no orphaned spaces or commas after deletions, a/an agreement ("an above-par" → "a below-par").
   - Unit and currency notation are restored to the article's style.
   - A sentence-final full stop dropped by the model is put back.
4. **Hint gating.** When hints exist, an edit not tagged with a hint is dropped. In evaluation every such unflagged edit was a false positive (for example, "defend 201" → "202").
5. **Repair calls (only when needed).**
   - If an edit's `find` cannot be located, the model is asked once to re-quote it.
   - If a flagged error span is still present verbatim, for instance because the error was repeated in a headline and only one copy was fixed, one targeted call fixes the remaining copies.
6. **Review pass, only for articles without hints** (001–003). A second, conservative check against the source catches misses. It is skipped when hints exist, because there it measurably caused over-edits.
7. **Graceful degradation.** Any failure falls back to the best text so far, at minimum the article with its annotation block removed. `rectify-all` always writes all 104 files.

### Evaluation (`evaluate.py`)

The 10 human references are copied to `eval/references/`, because `rectify-all` overwrites `rectified_articles/`. Metrics are word-level diffs of AI → reference (gold edits) against AI → ours:

| Metric | Value |
| --- | --- |
| Precision: our edits that land on a span the human also changed | ~96% |
| Recall: gold edits reproduced exactly, word for word | ~60% |
| Flagged error spans removed, across all 101 annotated articles | 100% (367/367) |

Most of the recall gap is wording that cannot be inferred. In article 004 the human rewrote whole sentences in the original phrasing, and in 001 they also reverted stylistic changes such as "unveiled" → "has launched". These are not factual errors.

### Key assumptions and design choices

- **The annotation block is a hint, not ground truth.** The reference diffs show the annotations accurately locate the injected errors and usually contain the original value. Their "corrections" are sometimes explanations rather than replacement text, so every hint is checked against the source and the source supplies the value.
- **"Not in the source" is not an error.** The generated articles include legitimate background knowledge. An early version deleted whole correct sections (article 006) until this rule was enforced in both the prompt and code-level size guards.
- **The `litellm` SDK was replaced by plain `requests`.** The pinned `litellm==1.49.5` is no longer installable from PyPI, and the proxy speaks the standard OpenAI `/chat/completions` protocol. `LLM_MODEL_NAME` accepts LiteLLM-style names (`openai/gpt-oss-120b`, `groq/openai/gpt-oss-120b`); the client tries the name and its provider-stripped suffixes until the endpoint accepts one.
- **Robustness.**
  - Retries with backoff on 429/5xx and on Groq's `json_validate_failed`, falling back to non-JSON mode if JSON mode keeps failing.
  - UTF-8-safe console output, so `✓`/`✗` can't crash a Windows cp1252 console.
  - Per-article traces of every applied, failed and rejected edit are written to `logs/`.
- **Tried and rejected.** Lower temperature made no measurable difference. `reasoning_effort=high` was no better than `medium` and cost more.

### Files

| File | Purpose |
| --- | --- |
| `rectifier.py` | CLI entry point (`test`, `rectify-all`): parallel processing, fallbacks, traces |
| `rectification_system.py` | Parsing, prompts, edit application, repair and review passes |
| `llm_client.py` | OpenAI-compatible client: retries, model-name resolution, token accounting |
| `evaluate.py` | Scores against the references, plus hint-coverage check over all articles |
| `eval/references/` | Copy of the 10 human-rectified examples |

---

## Problem Statement

Large Language Models (LLMs) are excellent at fluency and structure but notorious for factual inconsistency. We have a dataset of **104 articles** generated by an AI agent based on ground-truth source documents. While the writing style is high-quality, the content contains subtle but critical inaccuracies.

The source article used to generate each article is available in the `source_articles/` folder, and the respective article generated by the AI agent is available in the `ai_generated_articles/` folder.

**Your Mission**: Develop an automated **AI Editor** pipeline. Your system must intake an imperfect AI-generated article, cross-reference it with the original Source Text, and output a rectified version.

### The Core Challenge: Accuracy vs. Preservation

The difficulty of this challenge lies in the competing constraints:

1. **Factuality**: You must fix 100% of the hallucinations and errors.
2. **Preservation**: You must barely touch the text.

Most LLMs, when asked to "fix" a text, will rewrite the entire paragraph, changing the tone, sentence structure, and phrasing. **This is strictly forbidden**. Your system must act like a surgeon, not an author. It should excise the error and stitch in the fact without leaving a scar on the surrounding prose.

### Example

**AI-Generated Content (with error):**
> "The Pacific Ocean is the largest ocean, covering approximately 63 million square kilometers of Earth's surface."

**Error:** The Pacific Ocean covers approximately 165 million square kilometers, not 63 million.

**Version 1:**
> "The Pacific Ocean is the biggest ocean, covering approximately 165 million square kilometers of Earth's surface."

❌ **Why it's wrong:** The error is fixed, but other correct parts were unnecessarily changed ("largest" → "biggest").

**Version 2:**
> "The Pacific Ocean is the largest ocean, covering approximately 165 million square kilometers of Earth's surface."

✅ **Why it's correct:** Only the error was fixed ("63 million" → "165 million"), with no other modifications to the already-correct content.

### The Dataset

You are provided with three directories:

- `source_articles/`: This folder contains the source article used to generate each article.
- `ai_generated_articles/`: The article containing errors.
- `rectified_articles/`: 10 examples showing exactly how a human editor corrected the AI text. Use these to understand the expected output format and "strictness."

## LLM API Access & Budget Constraints

We will provide you with an LLM API key to develop and deploy your **AI Editor** pipeline.

**The Budget**: You have a strict hard limit of $3.00 USD for `openai/gpt-oss-120b` model.

Based on current model pricing, this equates to approximately **10 million tokens**. This budget is sufficient, but finite. You must manage this resource effectively to cover your development, testing, and internal validation.

### Monitoring Your Spend

You are responsible for tracking your usage. We have provided a `budget_checker.py` module to help you view your remaining balance in real-time.

```bash
python budget_checker.py
```

## Expectations & Deliverables

To ensure your submission can be automatically graded, you must adhere to the following strict operational requirements.

### 1. Reproducibility via Command Line

Your system must be fully executable via the provided entry point. We will grade your submission by running exactly **one** command:

```bash
python3 rectifier.py rectify-all
```

**Requirement**: When this command finishes, the `rectified_articles/` folder must contain **all 104** rectified text files. If your code requires manual intervention, notebook cells, or complex setups to generate the files, it will be marked as incomplete.

### 2. Dependency Management

You must ensure the environment is reproducible.

- **Update** `requirements.txt`: If you use any libraries not already listed (e.g., spacy, nltk, langchain, pydantic), you **must** add them to `requirements.txt`.
- **Standard Libraries**: Do not rely on libraries that require complex system-level installs (like specific C++ compilers) unless absolutely necessary and documented.

### 3. Documentation of Third-Party Services

If your solution relies on any external APIs or third-party services (other than the provided LiteLLM key), you must strictly document them in your `README.md`.

- Example: If you use a vector database (like Pinecone) or a search API, you must explain how to set it up and provide the necessary environment variable keys in a separate `env.example` file.

### 4. Output Integrity

Your code must save the final output to the rectified_articles/ directory.

- **File Naming**: The filenames must match the original IDs (e.g., article_001.txt).
- **Content**: The files should contain only the rectified article text. Do not include markdown code blocks (like ```) or JSON artifacts in the final text files.

## Getting Started

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure LLM credentials in a `.env` file (copy `env.example` to `.env` and fill in the values):

   | Variable | Description | Example |
   | --- | --- | --- |
   | `LLM_API_KEY` | API key for the LLM provider | `gsk_...` |
   | `LLM_API_BASE` | Base URL of the provider's OpenAI-compatible endpoint | `https://api.groq.com/openai/v1` |
   | `LLM_MODEL_NAME` | LiteLLM model identifier used for rectification | `groq/openai/gpt-oss-120b` |

   > **Note:** If you received the problem statement **along with the API key via direct email**, use the dedicated proxy endpoint and the API key provided in that email instead:
   >
   > ```bash
   > LLM_API_KEY=<the API key from the email>
   > LLM_API_BASE="https://recllm.brahmastra.tech/"
   > LLM_MODEL_NAME=openai/gpt-oss-120b
   > ```

3. Understand the codebase structure by reviewing `rectifier.py`

### Running the Demo

```bash
# Check remaining budget
python budget_checker.py

# Check budget with usage guide
python budget_checker.py --guide

# Test on first 16 articles
python rectifier.py test

# Test on custom count
python rectifier.py test --count 5

# Process all 104 articles
python rectifier.py rectify-all
```

## Building Your Solution

### Integration Point

Plug your rectification logic into `rectifier.py` at line 51:

```python
def rectify_article(article_id: str):
    ai_generated_content = get_ai_generated_article(article_id)
    
    # PLUG YOUR CUSTOM RECTIFIER HERE
    rectified_content = run(ai_generated_content)
    ###################################
    
    save_rectified_article(article_id, rectified_content)
    return rectified_content
```

### Implementation Freedom

You have complete flexibility to design your system:
- Replace the `run()` function or build a new architecture
- Access source articles via `get_article_mapping(article_id)`
- Create multi-file systems with any structure
- Use multiple LLM calls, validation layers, or confidence scoring
- Implement any approach that effectively solves the problem

## What to Submit

1. **Complete source code** with clear documentation
2. **Updated `requirements.txt`** with all dependencies
3. Documentation: A concise README explaining your approach and system architecture.

## ⚠️ Submission Verification (Critical)

We will evaluate your submission by running the following command on our environment:

```bash
python rectifier.py rectify-all
```

**If this command fails, crashes, or requires manual intervention, your submission will be disqualified.**

Ensure that your code handles exceptions gracefully and that your logic is robust enough to process the entire batch without stopping.

## Success Tips

- **Design your evaluation metric first**—use it to iteratively improve your system
- **Start small** (5-10 articles), validate, then scale up
- **Use the 10 reference examples** in `rectified_articles/` to validate your approach
- **Optimize token usage**—you have a limited budget (~10M tokens for development)
- **Document your reasoning** for key design decisions


---

**Good luck! We're excited to see your approach to this challenge.**

