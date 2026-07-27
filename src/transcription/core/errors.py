from __future__ import annotations


class TranscriptionError(Exception):
    """Base class for failures the user can act on.

    Interfaces (CLI, bot, web) catch this and show `str(exc)` directly, so
    messages are written for a user, not for a stack trace. Anything that
    isn't one of these is a bug and should keep its traceback.
    """


class FfmpegNotFoundError(TranscriptionError):
    pass


class AudioError(TranscriptionError):
    """ffmpeg/ffprobe rejected the file or failed to process it."""


class ApiError(TranscriptionError):
    """The transcription API refused the request or stayed unreachable."""


class ConfigError(TranscriptionError):
    """The config file or the resulting parameters are unusable."""
