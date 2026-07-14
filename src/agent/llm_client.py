import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, NotFoundError, APIStatusError

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://openrouter.ai/api/v1"
)
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
LLM_FORMAT_MODEL = os.getenv("LLM_FORMAT_MODEL", "meta-llama/llama-3.2-3b-instruct:free")

FREE_FALLBACK_MODELS = [
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]


def get_client() -> OpenAI:
    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=15,
        max_retries=0,
    )


def chat(messages: list[dict], model: str | None = None, model_key: str | None = None) -> str:
    client = get_client()
    if model_key == "format":
        primary = model or LLM_FORMAT_MODEL
    else:
        primary = model or LLM_MODEL
    models_to_try = [primary] + [
        m for m in FREE_FALLBACK_MODELS if m != primary
    ]

    last_error = None
    for attempt_model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=attempt_model,
                messages=messages,
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
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
