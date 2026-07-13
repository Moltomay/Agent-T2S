import os
import time
import re as _re
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://openrouter.ai/api/v1"
)
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

FREE_FALLBACK_MODELS = [
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "qwen/qwen-2.5-72b-instruct:free",
]


def get_client() -> OpenAI:
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def chat(messages: list[dict], model: str | None = None) -> str:
    client = get_client()
    primary = model or LLM_MODEL
    models_to_try = [primary] + [
        m for m in FREE_FALLBACK_MODELS if m != primary
    ]

    last_error = None
    for attempt_model in models_to_try:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=attempt_model,
                    messages=messages,
                    temperature=0.1,
                )
                return response.choices[0].message.content or ""
            except RateLimitError as e:
                last_error = e
                meta = str(e)
                match = _re.search(r"retry_after_seconds[\"':]+\s*(\d+)", meta)
                if match and int(match.group(1)) < 30:
                    time.sleep(int(match.group(1)) + 1)
                    continue
                break
            except Exception:
                raise

    raise last_error or Exception("All models rate-limited")
