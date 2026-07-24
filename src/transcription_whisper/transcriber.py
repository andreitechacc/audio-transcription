from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)


def transcribe_file(client: OpenAI, path: Path, language: str, model: str) -> str:
    logger.info("Transcribing %s (language=%s)", path, language)
    with open(path, "rb") as f:
        response = client.audio.transcriptions.with_raw_response.create(
            model=model,
            file=f,
            language=language,
        )
    request_id = response.headers.get("x-request-id", "unknown")
    logger.info("Request %s succeeded for %s", request_id, path)
    return response.parse().text
