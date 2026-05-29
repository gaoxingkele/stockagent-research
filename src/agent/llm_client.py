"""LLM client wrapper for Cloubic API (OpenAI-compatible).

Loads credentials from `.env.cloubic` (auto-discovered from repo root).
Supports Gemini 3.5 Flash (primary, cheap+fast) and Claude Opus 4.8 (pivotal review).

Usage:
    from src.agent.llm_client import call_llm
    response = call_llm(system="...", user="...", model="gemini-3.5-flash")
    print(response.content)   # str
    print(response.usage)     # {"input_tokens": ..., "output_tokens": ..., "cost_usd": ...}

Auto-retries on transient errors with exponential backoff.
"""
from __future__ import annotations
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env.cloubic"

# Load .env.cloubic on import
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=False)
else:
    logger.warning(f"No .env.cloubic found at {ENV_FILE}")

# Per-model pricing (USD per 1M tokens, approximate Cloubic pricing as of 2026-05)
PRICING = {
    # Gemini family
    "gemini-3.5-flash":            (0.075, 0.30),
    "gemini-3-flash-preview":      (0.075, 0.30),
    "gemini-3-pro":                (2.0, 10.0),
    # Claude family (1M context)
    "claude-opus-4-8":             (15.0, 75.0),
    "claude-opus-4-7":             (15.0, 75.0),
    "claude-sonnet-4-6":           (3.0, 15.0),
    "claude-haiku-4-5":            (1.0, 5.0),
    # GPT family
    "gpt-5.2":                     (5.0, 20.0),
    "gpt-5-mini":                  (0.30, 1.50),
    # DeepSeek
    "deepseek-v3":                 (0.27, 1.10),
}


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    raw: object | None = None  # raw API response if needed


def _make_client() -> OpenAI:
    base_url = os.getenv("CLOUBIC_BASE_URL") or "https://api.cloubic.com/v1"
    # Strip OpenAI SDK-appended path suffixes so we don't get /v1/chat/completions/chat/completions
    for tail in ("/chat/completions", "/completions"):
        if base_url.endswith(tail):
            base_url = base_url[: -len(tail)]
    base_url = base_url.rstrip("/")
    api_key = os.getenv("CLOUBIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"CLOUBIC_API_KEY not set. Ensure {ENV_FILE} exists and contains key."
        )
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model in PRICING:
        in_price, out_price = PRICING[model]
    else:
        # Default fallback: Gemini Flash pricing (cheapest)
        in_price, out_price = PRICING["gemini-3.5-flash"]
    return input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price


def call_llm(
    system: str,
    user: str,
    *,
    model: str = "gemini-3.5-flash",
    temperature: float = 0.2,
    max_tokens: int = 600,
    max_retries: int = 3,
    response_json_mode: bool = False,
) -> LLMResponse:
    """Send a chat completion request through Cloubic.

    Args:
        system: system prompt
        user: user prompt
        model: model ID (e.g. "gemini-3.5-flash", "claude-opus-4-8")
        temperature: sampling temperature (low for evaluation)
        max_tokens: max response tokens
        max_retries: exponential backoff retries on transient errors
        response_json_mode: request JSON object mode if supported

    Returns:
        LLMResponse with content, model, tokens, cost
    """
    client = _make_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(**kwargs)
            elapsed = time.time() - t0

            content = resp.choices[0].message.content or ""
            usage = resp.usage
            input_t = getattr(usage, "prompt_tokens", 0) if usage else 0
            output_t = getattr(usage, "completion_tokens", 0) if usage else 0
            cost = _estimate_cost(model, input_t, output_t)

            logger.debug(
                f"LLM call OK | model={model} elapsed={elapsed:.1f}s "
                f"in={input_t} out={output_t} cost=${cost:.5f}"
            )
            return LLMResponse(
                content=content, model=model,
                input_tokens=input_t, output_tokens=output_t,
                cost_usd=cost, raw=resp,
            )
        except (APIConnectionError, RateLimitError) as e:
            last_exc = e
            wait = 2 ** attempt
            logger.warning(f"Transient error (attempt {attempt}/{max_retries}): {e}. Retry in {wait}s")
            time.sleep(wait)
        except APIError as e:
            last_exc = e
            wait = 2 ** attempt
            logger.warning(f"API error (attempt {attempt}/{max_retries}): {e}. Retry in {wait}s")
            time.sleep(wait)
        except Exception as e:
            logger.error(f"Non-retryable error: {type(e).__name__}: {e}")
            raise
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_exc}")


def ping(model: str = "gemini-3.5-flash") -> bool:
    """Quick connectivity test."""
    try:
        r = call_llm(
            system="You are a helper.",
            user="Reply with the single word OK.",
            model=model,
            max_tokens=10,
            max_retries=1,
        )
        logger.info(f"Ping {model} OK: '{r.content.strip()}' (in={r.input_tokens} out={r.output_tokens})")
        return True
    except Exception as e:
        logger.error(f"Ping {model} FAILED: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=== Cloubic LLM client smoke ping ===")
    print(f"Base URL: {os.getenv('CLOUBIC_BASE_URL', '(default)')}")
    print(f"Key length: {len(os.getenv('CLOUBIC_API_KEY', ''))}")
    print()
    for model in ["gemini-3.5-flash"]:
        ok = ping(model)
        print(f"  {model}: {'OK' if ok else 'FAIL'}")
