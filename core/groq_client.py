"""Shared Groq API client utilities used across all agents."""

import os
import time
from typing import Any

from groq import Groq


def get_client() -> Groq:
    """Return a Groq client from the GROQ_API_KEY environment variable."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)


def default_model() -> str:
    """Return the model name, respecting the eval-override env var."""
    return os.environ.get("SHARP_RAG_EVAL_MODEL", "llama-3.3-70b-versatile")


def call_groq(
    client: Groq,
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    max_retries: int = 3,
    caller: str = "Groq",
) -> Any:
    """Call the Groq chat completions API with exponential-backoff retry.

    Parameters
    ----------
    client:       A Groq client instance.
    system_prompt: System message content.
    user_message:  User message content.
    model:        Model ID; defaults to ``default_model()`` if None.
    temperature:  Sampling temperature (default 0.2).
    max_tokens:   Max tokens in the completion (default 512).
    max_retries:  Number of attempts before re-raising (default 3).
    caller:       Label printed in retry logs (e.g. "CritiqueAgent").

    Returns
    -------
    The raw Groq chat completions response object.
    """
    if model is None:
        model = default_model()

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[{caller}] Groq API error (attempt {attempt + 1}): {exc}. "
                  f"Retrying in {wait}s…")
            time.sleep(wait)
