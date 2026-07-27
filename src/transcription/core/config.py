from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from .errors import ConfigError

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.toml")

# Formats the API returns as a plain body rather than a JSON object.
RAW_TEXT_FORMATS = frozenset({"text", "srt", "vtt"})
KNOWN_FORMATS = RAW_TEXT_FORMATS | {"json", "verbose_json", "diarized_json"}


@dataclass(frozen=True)
class TranscriptionParams:
    """Everything one transcription run needs to know.

    This is deliberately a plain value object rather than "the config": it is
    built from a file for the CLI, but the Telegram bot will build one per
    message and the web API one per job, without a file anywhere. Keeping the
    file out of it is what lets the same pipeline serve all three.
    """

    language: str = "en"
    model: str = "gpt-4o-transcribe-diarize"
    response_format: str = "diarized_json"
    max_size_mb: float = 24.0
    bitrate: str = "64k"
    # An upper bound, not a target. Splitting costs quality — the API keeps
    # speaker labels consistent within one request but assigns them afresh in
    # the next, so every split is a seam where "A" may become "B". Sending as
    # much audio per request as the size limit allows is therefore the better
    # default, and at 64k mono that limit is reached at roughly 50 minutes.
    chunk_minutes: int = 45
    # Transcribing 45 minutes in one request takes several minutes of wall
    # clock time, during which the connection carries nothing. The SDK default
    # is far too short for that.
    request_timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        # Validating here rather than in the loader means CLI flags and bot
        # commands get the same checks for free — there is no way to build an
        # invalid params object.
        if self.response_format not in KNOWN_FORMATS:
            raise ConfigError(
                f"Unknown response_format {self.response_format!r}. "
                f"Expected one of: {', '.join(sorted(KNOWN_FORMATS))}."
            )
        if self.response_format == "diarized_json" and "diarize" not in self.model:
            # Caught here because the API's own error for this is opaque.
            raise ConfigError(
                f"response_format 'diarized_json' requires a diarizing model, "
                f"but model is {self.model!r}. Use 'gpt-4o-transcribe-diarize', "
                f"or switch response_format to 'json'."
            )
        if self.chunk_minutes < 1:
            raise ConfigError("chunk_minutes must be at least 1.")
        if self.max_size_mb <= 0:
            raise ConfigError("max_size_mb must be greater than 0.")
        if self.request_timeout_seconds <= 0:
            raise ConfigError("request_timeout_seconds must be greater than 0.")

    @property
    def chunk_separator(self) -> str:
        """What to put between two chunks' text when joining them.

        Diarized output is line-oriented, so chunks are separated by a blank
        line; plain transcripts are a single flowing text, so a space is
        enough. Lives here so that every interface joins chunks identically.
        """
        return "\n\n" if self.response_format == "diarized_json" else " "

    @property
    def is_diarized(self) -> bool:
        return self.response_format == "diarized_json"


def load_params(path: Path | None = None) -> TranscriptionParams:
    """Build params from a TOML file, falling back to the built-in defaults."""
    path = path or DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning("Config file %s not found, using built-in defaults", path)
        return TranscriptionParams()

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc

    known = {f.name for f in fields(TranscriptionParams)}
    unknown = set(data) - known
    if unknown:
        # A typo in a config key would otherwise be silently ignored, and the
        # run would quietly use a default the user thought they had changed.
        logger.warning(
            "Ignoring unknown key(s) in %s: %s", path, ", ".join(sorted(unknown))
        )

    values = {key: data[key] for key in known & set(data)}
    try:
        return TranscriptionParams(**values)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid value in {path}: {exc}") from exc
