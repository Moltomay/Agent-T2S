"""OpenAI-compatible LLM wrapper with fallback chain across providers.

Supports native function calling (``tools`` parameter). Returns a
``Message`` object when ``tools`` is provided, raw string otherwise.
Falls back through a list of free-tier models on 429 rate limits.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, NotFoundError, APIStatusError

load_dotenv()

LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
LLM_FORMAT_MODEL: str = os.getenv("LLM_FORMAT_MODEL", "meta-llama/llama-3.2-3b-instruct:free")

FREE_FALLBACK_MODELS: list[str] = [
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]


def get_client() -> OpenAI:
    """Return a configured OpenAI client targeting the configured base URL."""
    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=15,
        max_retries=0,
    )


def chat(
    messages: list[dict],
    model: str | None = None,
    model_key: str | None = None,
    tools: list[dict] | None = None,
) -> str | object:
    """Send a chat completion request and return the response.

    Args:
        messages: Standard OpenAI message list.
        model: Override the default model name.
        model_key: If ``"format"``, uses the cheaper ``LLM_FORMAT_MODEL`` (for memory summarisation).
        tools: Function-calling tool definitions. When present, returns the
            full ``Message`` object (with ``tool_calls``) instead of a string.

    Returns:
        ``Message`` object if ``tools`` is provided, else the content string.

    Raises:
        Exception: All providers exhausted (all rate-limited or unavailable).
    """
    client = get_client()
    if model_key == "format":
        primary: str = model or LLM_FORMAT_MODEL
    else:
        primary = model or LLM_MODEL
    models_to_try: list[str] = [primary] + [m for m in FREE_FALLBACK_MODELS if m != primary]

    last_error: Exception | None = None
    for attempt_model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=attempt_model,
                messages=messages,
                tools=tools,
                temperature=0.1,
            )
            msg = response.choices[0].message
            if tools:
                return msg
            return msg.content or ""
        except RateLimitError as e:
            last_error = e
            continue
        except NotFoundError:
            continue
        except APIStatusError as e:
            if e.status_code == 429:
                last_error = e
                continue
            raise
        except Exception:
            raise

    raise last_error or Exception("All LLM providers rate-limited. Try again later.")
