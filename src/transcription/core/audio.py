from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from .errors import AudioError, FfmpegNotFoundError

logger = logging.getLogger(__name__)

# Nothing here decodes audio in Python. Every operation is a single ffmpeg
# invocation that streams the file, so memory use stays flat no matter how
# long the recording is, and the audio is decoded once per operation instead
# of once per Python-level access to it.


def ensure_ffmpeg() -> None:
    """Fail early, with a message that says what to install.

    Checked once at startup rather than at first use, so a missing ffmpeg is
    reported before we spend time reading config or uploading anything.
    """
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise FfmpegNotFoundError(
            f"{' and '.join(missing)} not found on PATH. "
            "Install ffmpeg (https://ffmpeg.org/download.html) and make sure "
            "the directory containing these executables is on PATH."
        )


def _run(command: list[str], what: str) -> str:
    """Run an ffmpeg-family command, turning a non-zero exit into AudioError."""
    logger.debug("Running: %s", " ".join(command))
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:  # executable vanished between ensure_ffmpeg() and now
        raise AudioError(f"Could not run {command[0]}: {exc}") from exc

    if completed.returncode != 0:
        # ffmpeg's diagnostics go to stderr and the useful part is at the end;
        # the first lines are just build configuration banners.
        tail = "\n".join(completed.stderr.strip().splitlines()[-5:])
        raise AudioError(
            f"{what} failed (ffmpeg exit code {completed.returncode}):\n{tail}"
        )
    return completed.stdout


def get_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def probe_duration_minutes(path: Path) -> float:
    """Read duration from container metadata without decoding the audio.

    This is the main reason to call ffprobe instead of loading the file: the
    answer comes from the file header in milliseconds, whereas decoding a
    two-hour recording just to measure it would take seconds and hundreds of
    megabytes of RAM.
    """
    output = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        f"Reading duration of {path.name}",
    ).strip()

    try:
        return float(output) / 60
    except ValueError as exc:
        # Streams with no duration in the header (some live recordings) print
        # "N/A" and exit 0, so this is a normal outcome, not a crash.
        raise AudioError(
            f"Could not determine the duration of {path.name} — the file may be "
            f"corrupt or not an audio file (ffprobe reported {output!r})."
        ) from exc


def _encode_args(bitrate: str) -> list[str]:
    """Shared encoder settings for every mp3 we produce.

    -vn drops cover art: phone and podcast files often carry an embedded
    image as a video stream, and the mp3 muxer would otherwise try to write
    it and fail. -ac 1 downmixes to mono, which for speech costs nothing in
    accuracy and halves the bytes uploaded at a given bitrate.
    """
    return ["-vn", "-ac", "1", "-b:a", bitrate]


def transcode_to_mp3(input_path: Path, output_path: Path, bitrate: str) -> Path:
    logger.info(
        "Transcoding %s -> %s (bitrate=%s)", input_path.name, output_path.name, bitrate
    )
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(input_path),
            *_encode_args(bitrate),
            str(output_path),
        ],
        f"Transcoding {input_path.name}",
    )
    return output_path


def split_to_mp3_chunks(
    input_path: Path, out_dir: Path, chunk_seconds: float, bitrate: str
) -> list[Path]:
    """Cut the input into mp3 chunks of at most `chunk_seconds` each.

    Takes seconds rather than whole minutes so a recording can be divided into
    equal parts: splitting 29 minutes at a 23-minute limit gives 14.5 + 14.5
    rather than 23 + 6, and a short trailing fragment is the worst piece to
    hand a diarizer.

    ffmpeg's segment muxer does the whole split in one pass: it decodes the
    input once and writes every chunk as it goes, rather than us seeking back
    into the file separately for each chunk.
    """
    pattern = out_dir / "chunk_%03d.mp3"
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(input_path),
            *_encode_args(bitrate),
            "-f",
            "segment",
            "-segment_time",
            f"{chunk_seconds:.3f}",
            # Without this, each chunk keeps the original file's timeline, so
            # chunk 3 would claim to start at 20:00 and duration probes (and
            # some players) get confused.
            "-reset_timestamps",
            "1",
            str(pattern),
        ],
        f"Splitting {input_path.name}",
    )

    chunks = sorted(out_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise AudioError(f"Splitting {input_path.name} produced no chunks.")
    logger.info("Split %s into %d chunk(s)", input_path.name, len(chunks))
    return chunks
