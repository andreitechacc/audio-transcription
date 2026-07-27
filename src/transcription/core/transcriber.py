from __future__ import annotations

import logging
from pathlib import Path

import openai
from openai import OpenAI

from .config import TranscriptionParams
from .errors import ApiError

logger = logging.getLogger(__name__)

# The client's defaults are tuned for fast chat calls. A single transcription
# request legitimately takes a minute or more, and flaky links (VPNs, proxies)
# drop long-lived connections, so both numbers are raised well above them.
REQUEST_TIMEOUT_SECONDS = 120.0
MAX_RETRIES = 5


def build_client(api_key: str | None = None) -> OpenAI:
    """Create an API client with transcription-appropriate timeouts.

    Every interface builds its client through here so none of them silently
    fall back to the SDK's short default timeout.
    """
    try:
        return OpenAI(
            api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES
        )
    except openai.OpenAIError as exc:
        # Raised when no key is configured at all.
        raise ApiError(
            f"Could not create the OpenAI client: {exc}. "
            "Set OPENAI_API_KEY in your environment or .env file."
        ) from exc


def transcribe_audio(client: OpenAI, path: Path, params: TranscriptionParams) -> str:
    """Transcribe one audio file that is already within the API's limits."""
    logger.info(
        "Transcribing %s (language=%s, model=%s)", path.name, params.language, params.model
    )

    kwargs = {}
    if params.is_diarized:
        # Lets the API do its own long-audio segmentation before diarizing,
        # which keeps speaker labels consistent within a single request.
        kwargs["chunking_strategy"] = "auto"

    try:
        with open(path, "rb") as f:
            # with_raw_response keeps the HTTP headers reachable so the
            # request id can be logged; response.parse() then gives the same
            # object the normal call would have returned.
            response = client.audio.transcriptions.with_raw_response.create(
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
    except openai.RateLimitError as exc:
        raise ApiError(
            "OpenAI rate limit or quota exceeded. Check your billing and usage "
            "limits, then retry."
        ) from exc
    except openai.APIConnectionError as exc:
        raise ApiError(
            f"Could not reach OpenAI: {exc}. Check your connection — if you are "
            "using a VPN, note that some configurations drop long-running "
            "requests before the transcription finishes."
        ) from exc
    except openai.APIStatusError as exc:
        raise ApiError(
            f"OpenAI returned an error for {path.name} (HTTP {exc.status_code}): "
            f"{exc.message}"
        ) from exc

    request_id = response.headers.get("x-request-id", "unknown")
    logger.info("Request %s succeeded for %s", request_id, path.name)

    return _extract_text(response.parse(), params)


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


__all__ = [
    "MAX_RETRIES",
    "REQUEST_TIMEOUT_SECONDS",
    "build_client",
    "transcribe_audio",
]
