from __future__ import annotations

import logging
import math
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from .audio import (
    ensure_ffmpeg,
    get_size_mb,
    probe_duration_minutes,
    split_to_mp3_chunks,
    transcode_to_mp3,
)
from .config import TranscriptionParams
from .errors import TranscriptionError
from .transcriber import transcribe_audio

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Progress:
    """A progress update, in a form any interface can render.

    `total` is 0 while the chunk count is still unknown (before the audio has
    been inspected). The CLI prints `message`; the bot will edit its status
    message; the web API will store `current`/`total` for a progress bar.
    """

    stage: str  # "analyzing" | "preparing" | "transcribing" | "finished"
    current: int
    total: int
    message: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    chunks: tuple[str, ...]
    duration_minutes: float
    was_split: bool


ProgressCallback = Callable[[Progress], None]
# Called with (chunk_index, chunk_text) the moment a chunk comes back.
PartialCallback = Callable[[int, str], None]


def _notify(callback: ProgressCallback | None, progress: Progress) -> None:
    """Report progress without letting the reporting break the job.

    A transcription that already cost money must not be lost because the
    thing displaying it failed — for the CLI that is unlikely, but the bot
    reports progress by calling the Telegram API, which can rate-limit or
    time out at any moment.
    """
    if callback is None:
        return
    try:
        callback(progress)
    except Exception:
        logger.warning("Progress callback failed; continuing", exc_info=True)


def transcribe(
    input_path: Path,
    params: TranscriptionParams,
    client: OpenAI,
    *,
    on_progress: ProgressCallback | None = None,
    on_partial: PartialCallback | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file, splitting it first if necessary.

    Returns the transcript instead of writing it anywhere, and reports
    progress through callbacks instead of logging it for a specific UI. That
    is what lets the CLI, the Telegram bot and the web API share this code:
    each one decides for itself where the text goes and how progress is shown.
    """
    if not input_path.exists():
        raise TranscriptionError(f"Input file not found: {input_path}")
    ensure_ffmpeg()

    work_dir = Path(tempfile.mkdtemp(prefix="transcribe_"))
    try:
        _notify(on_progress, Progress("analyzing", 0, 0, "Inspecting audio..."))
        size_mb = get_size_mb(input_path)
        duration_minutes = probe_duration_minutes(input_path)
        logger.info("Input: %.1f MB, %.1f min", size_mb, duration_minutes)

        sources, was_split = _prepare_audio(
            input_path, work_dir, params, duration_minutes, size_mb, on_progress
        )
        chunks = _transcribe_all(
            sources, client, params, was_split, on_progress, on_partial
        )
    except Exception:
        # Kept rather than cleaned up: re-encoding a long file is slow, and on
        # a failure the intermediate mp3s are exactly what a retry needs.
        logger.error(
            "Transcription failed; keeping intermediate files for inspection at %s",
            work_dir,
        )
        raise
    else:
        shutil.rmtree(work_dir, ignore_errors=True)

    _notify(
        on_progress,
        Progress("finished", len(chunks), len(chunks), "Transcription complete."),
    )
    return TranscriptionResult(
        text=params.chunk_separator.join(chunks),
        chunks=tuple(chunks),
        duration_minutes=duration_minutes,
        was_split=was_split,
    )


def _prepare_audio(
    input_path: Path,
    work_dir: Path,
    params: TranscriptionParams,
    duration_minutes: float,
    size_mb: float,
    on_progress: ProgressCallback | None,
) -> tuple[list[Path], bool]:
    """Turn the input into one or more files the API will accept.

    Long audio is split even when it is under the size limit: a single
    request covering a long recording takes long enough that flaky links
    (proxies, some VPNs) drop the connection before the response arrives.
    Several short requests are individually far more likely to complete.
    """
    if duration_minutes <= params.max_request_minutes:
        _notify(on_progress, Progress("preparing", 0, 0, "Converting audio..."))
        # Re-encoding a short file is not only about size: some phone
        # recordings are .m4a with a container brand (e.g. 3gp4) that the
        # API's format check rejects even though m4a is nominally supported,
        # and a plain mp3 sidesteps that.
        normalized = transcode_to_mp3(
            input_path, work_dir / "normalized.mp3", params.bitrate
        )
        normalized_size = get_size_mb(normalized)
        logger.info("Converted size: %.1f MB", normalized_size)
        if normalized_size <= params.max_size_mb:
            return [normalized], False
        logger.info("Still over %.1f MB after conversion; splitting", params.max_size_mb)

    chunk_seconds = _plan_chunk_length(duration_minutes, params.max_request_minutes)
    logger.info(
        "Splitting %.1f min into %d chunks of %.1f min (size=%.1f MB)",
        duration_minutes,
        round(duration_minutes * 60 / chunk_seconds),
        chunk_seconds / 60,
        size_mb,
    )
    if params.is_diarized:
        # Each chunk is diarized independently, so speaker labels ("A", "B")
        # are not guaranteed to mean the same person across chunk boundaries.
        logger.warning(
            "Diarizing in separate chunks; speaker labels may not stay "
            "consistent across chunk boundaries"
        )
    _notify(on_progress, Progress("preparing", 0, 0, "Splitting audio..."))
    # Chunks are cut from the original file, not from a re-encoded copy: the
    # split re-encodes each chunk anyway, so converting first would double the
    # work and stack one lossy encode on another.
    chunks = split_to_mp3_chunks(input_path, work_dir, chunk_seconds, params.bitrate)
    return chunks, True


def _plan_chunk_length(duration_minutes: float, max_minutes: float) -> float:
    """Choose a chunk length in seconds: as few chunks as allowed, all equal.

    Fewer chunks means fewer seams where speaker labels can be reassigned, and
    equal chunks avoid leaving a short trailing fragment — the piece a
    diarizer has least context for. Splitting 29 minutes under a 23-minute
    limit therefore gives two chunks of 14.5, not 23 followed by 6.
    """
    chunk_count = math.ceil(duration_minutes / max_minutes)
    return duration_minutes * 60 / chunk_count


def _transcribe_all(
    sources: list[Path],
    client: OpenAI,
    params: TranscriptionParams,
    was_split: bool,
    on_progress: ProgressCallback | None,
    on_partial: PartialCallback | None,
) -> list[str]:
    total = len(sources)
    texts: list[str] = []

    for index, source in enumerate(sources):
        label = f"Transcribing part {index + 1} of {total}..." if was_split else "Transcribing..."
        _notify(on_progress, Progress("transcribing", index + 1, total, label))

        text = transcribe_audio(client, source, params)
        texts.append(text)

        # Handed over immediately rather than after the loop, so a failure on
        # a later chunk does not throw away results that already cost money.
        if on_partial is not None:
            on_partial(index, text)

    return texts
