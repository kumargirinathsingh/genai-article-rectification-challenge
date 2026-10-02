"""
Minimal OpenAI-compatible chat client (works against the LiteLLM proxy or Groq).

Plain `requests` is used instead of the `litellm` SDK: the originally pinned
litellm==1.49.5 is no longer installable from PyPI, and the proxy speaks the
standard OpenAI /chat/completions protocol anyway.
"""

import os
import random
import threading
import time

import requests
from dotenv import load_dotenv

load_dotenv()

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}
# Error markers meaning "this model name is not usable here" (try the next candidate name).
MODEL_REJECTION_MARKERS = ("team_model_access_denied", "model_not_found", "not allowed to access model",
                           "does not exist", "invalid model", "unknown model", "no such model")


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY", "")
        base = (os.getenv("LLM_API_BASE") or "").strip().strip('"').rstrip("/")
        if not base:
            raise LLMError("LLM_API_BASE is not set (see env.example)")
        self.url = base + "/chat/completions"
        self.model_candidates = self._model_candidates(os.getenv("LLM_MODEL_NAME") or "gpt-oss-120b")
        self._resolved_model = None
        self._lock = threading.Lock()
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    @staticmethod
    def _model_candidates(name: str):
        # LLM_MODEL_NAME may be a LiteLLM identifier such as "openai/gpt-oss-120b"
        # or "groq/openai/gpt-oss-120b", where leading segments name the provider.
        # The raw endpoint may want the full name or a suffix of it, so try each.
        parts = name.strip().strip('"').split("/")
        return list(dict.fromkeys("/".join(parts[i:]) for i in range(len(parts))))

    def chat(self, messages, *, reasoning_effort="medium", max_tokens=16000,
             json_mode=True, temperature=None, timeout=180, max_attempts=6):
        models = [self._resolved_model] if self._resolved_model else self.model_candidates
        last_err = None
        for model in models:
            payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
            if reasoning_effort:
                payload["reasoning_effort"] = reasoning_effort
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            if temperature is not None:
                payload["temperature"] = temperature
            try:
                data = self._post_with_retries(payload, timeout, max_attempts)
            except _ModelRejected as e:
                last_err = e
                continue
            self._resolved_model = model
            self._record_usage(data.get("usage") or {})
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            return content, choice.get("finish_reason")
        raise LLMError(f"No usable model among {models}: {last_err}")

    def _post_with_retries(self, payload, timeout, max_attempts):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = dict(payload)
        last_err = None
        json_failures = 0
        for attempt in range(max_attempts):
            try:
                resp = requests.post(self.url, json=payload, headers=headers, timeout=timeout)
            except requests.RequestException as e:
                last_err = e
            else:
                if resp.status_code == 200:
                    return resp.json()
                body = resp.text[:1000]
                low = body.lower()
                # Groq returns 400 json_validate_failed when JSON-mode output is malformed.
                # Retry the sample; after two such failures drop JSON mode (the caller
                # extracts the JSON object from free text).
                if "json_validate_failed" in low:
                    json_failures += 1
                    if json_failures >= 2:
                        payload.pop("response_format", None)
                    last_err = LLMError(f"HTTP {resp.status_code}: {body[:300]}")
                    continue
                if resp.status_code in (400, 403, 404) and any(m in low for m in MODEL_REJECTION_MARKERS):
                    raise _ModelRejected(f"{resp.status_code}: {body[:300]}")
                if resp.status_code not in RETRYABLE_STATUS:
                    raise LLMError(f"HTTP {resp.status_code}: {body[:300]}")
                last_err = LLMError(f"HTTP {resp.status_code}: {body}")
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        time.sleep(min(float(retry_after), 60))
                        continue
                    except ValueError:
                        pass
            time.sleep(min(2 ** attempt, 30) + random.random())
        raise LLMError(f"Request failed after {max_attempts} attempts: {last_err}")

    def _record_usage(self, usage):
        with self._lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += usage.get("completion_tokens", 0) or 0


class _ModelRejected(Exception):
    pass


_client = None
_client_lock = threading.Lock()


def get_client() -> LLMClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = LLMClient()
        return _client
