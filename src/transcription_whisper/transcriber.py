from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)


def transcribe_file(
    client: OpenAI, path: Path, language: str, model: str, response_format: str
) -> str:
    logger.info("Transcribing %s (language=%s, model=%s)", path, language, model)
    kwargs = {}
    if response_format == "diarized_json":
        # Lets the API do its own long-audio segmentation before diarizing,
        # which keeps speaker labels consistent within a single request.
        kwargs["chunking_strategy"] = "auto"

    with open(path, "rb") as f:
        response = client.audio.transcriptions.with_raw_response.create(
            model=model,
            file=f,
            language=language,
            response_format=response_format,
            **kwargs,
        )
    request_id = response.headers.get("x-request-id", "unknown")
    logger.info("Request %s succeeded for %s", request_id, path)

    parsed = response.parse()
    if isinstance(parsed, str):
        # response_format in {"text", "srt", "vtt"} return the raw body as-is.
        return parsed
    if response_format == "diarized_json":
        return "\n".join(f"[{seg.speaker}] {seg.text.strip()}" for seg in parsed.segments)
    return parsed.text
