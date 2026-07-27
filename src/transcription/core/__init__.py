"""Transcription logic, independent of any interface.

Nothing in this package knows about argparse, Telegram or HTTP. The CLI (and
later the bot and the web API) import from here and add only their own
input/output handling on top.
"""

from .config import KNOWN_FORMATS, TranscriptionParams, load_params
from .errors import (
    ApiError,
    AudioError,
    ConfigError,
    FfmpegNotFoundError,
    TranscriptionError,
)
from .pipeline import Progress, TranscriptionResult, transcribe
from .transcriber import build_client

__all__ = [
    "KNOWN_FORMATS",
    "ApiError",
    "AudioError",
    "ConfigError",
    "FfmpegNotFoundError",
    "Progress",
    "TranscriptionError",
    "TranscriptionParams",
    "TranscriptionResult",
    "build_client",
    "load_params",
    "transcribe",
]
