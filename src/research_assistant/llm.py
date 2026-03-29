import json
import logging
import re
import time
from typing import TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from research_assistant.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def call_llm(prompt: str, system: str, settings: Settings) -> str:
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    logger.debug("LLM call: system=%s..., prompt=%s...", system[:100], prompt[:100])

    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    logger.debug("LLM response: %s...", text[:200])
    logger.info(
        "LLM usage: input=%d output=%d",
        response.usage.input_tokens,
        response.usage.output_tokens,
    )
    return text


def parse_json_response(raw: str) -> dict | list:
    cleaned = raw.strip()
    # Strip markdown fences
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}\nRaw: {raw[:500]}") from e


def validate_against_schema(data: dict, model: type[T]) -> T:
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ValueError(
            f"Schema validation failed for {model.__name__}:\n{e}"
        ) from e


def retry_with_backoff(
    func,
    max_retries: int = 3,
    base: float = 1.0,
    factor: float = 4.0,
):
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            ValueError,
        ) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base * (factor ** attempt)
                logger.warning(
                    "Retry %d/%d after error: %s (waiting %.1fs)",
                    attempt + 1, max_retries, e, delay,
                )
                time.sleep(delay)
    raise last_error


def llm_call_with_validation(
    prompt: str,
    system: str,
    model: type[T],
    settings: Settings,
) -> T:
    error_context = ""

    def _attempt():
        nonlocal error_context
        full_prompt = prompt
        if error_context:
            full_prompt += (
                f"\n\nPrevious attempt failed validation:\n{error_context}\n"
                "Please fix the output to match the required schema."
            )
        raw = call_llm(full_prompt, system, settings)
        data = parse_json_response(raw)
        try:
            return validate_against_schema(data, model)
        except ValueError as e:
            error_context = str(e)
            raise

    return retry_with_backoff(
        _attempt,
        max_retries=settings.llm_max_retries,
        base=settings.llm_backoff_base,
        factor=settings.llm_backoff_factor,
    )


def llm_call_with_list_validation(
    prompt: str,
    system: str,
    model: type[T],
    settings: Settings,
) -> list[T]:
    error_context = ""

    def _attempt():
        nonlocal error_context
        full_prompt = prompt
        if error_context:
            full_prompt += (
                f"\n\nPrevious attempt failed validation:\n{error_context}\n"
                "Please fix the output to match the required schema."
            )
        raw = call_llm(full_prompt, system, settings)
        data = parse_json_response(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")
        results = []
        errors = []
        for i, item in enumerate(data):
            try:
                results.append(validate_against_schema(item, model))
            except ValueError as e:
                errors.append(f"Item {i}: {e}")
        if errors and not results:
            error_context = "\n".join(errors)
            raise ValueError(f"All items failed validation:\n{error_context}")
        if errors:
            logger.warning("Some items failed validation: %s", "\n".join(errors))
        return results

    return retry_with_backoff(
        _attempt,
        max_retries=settings.llm_max_retries,
        base=settings.llm_backoff_base,
        factor=settings.llm_backoff_factor,
    )
