from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.toml")


@dataclass(frozen=True)
class Config:
    language: str
    model: str
    response_format: str
    max_size_mb: float
    bitrate: str
    chunk_minutes: int

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or DEFAULT_CONFIG_PATH
        if not path.exists():
            logger.warning("Config file %s not found, using built-in defaults", path)
            data = {}
        else:
            with open(path, "rb") as f:
                data = tomllib.load(f)

        return cls(
            language=data.get("language", "en"),
            model=data.get("model", "whisper-1"),
            response_format=data.get("response_format", "json"),
            max_size_mb=float(data.get("max_size_mb", 24)),
            bitrate=data.get("bitrate", "64k"),
            chunk_minutes=int(data.get("chunk_minutes", 10)),
        )
