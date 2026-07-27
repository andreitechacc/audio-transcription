from __future__ import annotations

import logging
import time
from pathlib import Path

import openai
from openai import OpenAI

from .config import TranscriptionParams
from .errors import ApiError

logger = logging.getLogger(__name__)

# Retries are handled here rather than by the SDK. The SDK retries connection
# errors and timeouts too, and those are exactly the failures that must NOT be
# retried automatically: the audio was uploaded, the server transcribed it and
# billed for it, and only the response was lost. Retrying pays for the same
# work again with no better odds of the response arriving.
#
# What is safe to retry is a request the server refused before doing any work:
# a rate limit, or a 5xx.
RETRYABLE_STATUS_CODES = frozenset({500, 502, 503, 504})
MAX_SAFE_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2.0


def build_client(
    api_key: str | None = None, timeout_seconds: float = 900.0
) -> OpenAI:
    """Create an API client configured for long transcription requests.

    The SDK's defaults are tuned for fast chat calls; transcribing half an
    hour of audio in one request takes several minutes. max_retries is zero
    because retrying is decided in transcribe_audio, where it can tell a free
    failure from one that has already been billed.
    """
    try:
        return OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
    except openai.OpenAIError as exc:
        # Raised when no key is configured at all.
        raise ApiError(
            f"Could not create the OpenAI client: {exc}. "
            "Set OPENAI_API_KEY in your environment or .env file."
        ) from exc


def transcribe_audio(client: OpenAI, path: Path, params: TranscriptionParams) -> str:
    """Transcribe one audio file that is already within the API's limits."""
    logger.info(
        "Transcribing %s (language=%s, model=%s)",
        path.name,
        params.language,
        params.model,
    )

    kwargs = {}
    if params.is_diarized:
        # Lets the API do its own segmentation before diarizing, which keeps
        # speaker labels consistent for the whole of this request. It is the
        # reason to send as much audio per request as the limits allow.
        kwargs["chunking_strategy"] = "auto"

    response = _create_with_retries(client, path, params, kwargs)

    request_id = response.headers.get("x-request-id", "unknown")
    logger.info("Request %s succeeded for %s", request_id, path.name)

    return _extract_text(response.parse(), params)


def _create_with_retries(
    client: OpenAI, path: Path, params: TranscriptionParams, kwargs: dict
):
    """Send the request, retrying only failures that cost nothing."""
    for attempt in range(MAX_SAFE_RETRIES + 1):
        try:
            with open(path, "rb") as f:
                # with_raw_response keeps the HTTP headers reachable so the
                # request id can be logged; response.parse() then returns the
                # same object a normal call would have.
                return client.audio.transcriptions.with_raw_response.create(
                    model=params.model,
                    file=f,
                    language=params.language,
                    response_format=params.response_format,
                    **kwargs,
                )
        except openai.AuthenticationError as exc:
            raise ApiError(
                "OpenAI rejected the API key. Check OPENAI_API_KEY in your .env file."
            ) from exc
        except openai.APITimeoutError as exc:
            # Must be caught before APIConnectionError, which it subclasses —
            # otherwise a timeout is reported as a connection problem and sends
            # you looking at your network instead of at the timeout setting.
            raise ApiError(
                f"OpenAI did not respond within the timeout while transcribing "
                f"{path.name}. The audio was most likely transcribed and billed, "
                f"and only the response was lost. Raise request_timeout_seconds "
                f"in config.toml, or lower chunk_minutes so each request covers "
                f"less audio."
            ) from exc
        except openai.APIConnectionError as exc:
            raise ApiError(
                f"Could not reach OpenAI while transcribing {path.name}: {exc}. "
                "If you are on a VPN, note that some configurations close "
                "connections that stay idle, which a long transcription does."
            ) from exc
        except openai.RateLimitError as exc:
            _wait_before_retry(attempt, "rate limited", exc)
        except openai.APIStatusError as exc:
            if exc.status_code not in RETRYABLE_STATUS_CODES:
                raise ApiError(
                    f"OpenAI returned an error for {path.name} "
                    f"(HTTP {exc.status_code}): {exc.message}"
                ) from exc
            _wait_before_retry(attempt, f"server error {exc.status_code}", exc)

    raise AssertionError("unreachable")  # pragma: no cover


def _wait_before_retry(attempt: int, reason: str, exc: Exception) -> None:
    """Sleep before the next attempt, or give up if they are exhausted.

    Only reached for failures where the server did no work, so a retry is
    free apart from the waiting.
    """
    if attempt >= MAX_SAFE_RETRIES:
        raise ApiError(
            f"OpenAI kept failing ({reason}) after {MAX_SAFE_RETRIES + 1} attempts: {exc}"
        ) from exc

    delay = RETRY_BACKOFF_SECONDS * (2**attempt)
    logger.warning("Request %s; retrying in %.0fs", reason, delay)
    time.sleep(delay)


def _extract_text(parsed: object, params: TranscriptionParams) -> str:
    """Reduce whatever the API returned to plain text.

    The shape depends on response_format, so the branching lives in one place
    instead of at every call site.
    """
    if isinstance(parsed, str):
        # text/srt/vtt come back as the raw body.
        return parsed
    if params.is_diarized:
        return "\n".join(
            f"[{segment.speaker}] {segment.text.strip()}" for segment in parsed.segments
        )
    return parsed.text


__all__ = ["MAX_SAFE_RETRIES", "build_client", "transcribe_audio"]
