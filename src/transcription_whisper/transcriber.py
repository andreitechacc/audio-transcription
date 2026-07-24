from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)


def transcribe_file(client: OpenAI, path: Path, language: str, model: str) -> str:
    logger.info("Transcribing %s (language=%s)", path, language)
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=model,
            file=f,
            language=language,
        )
    return result.text
