from __future__ import annotations

import logging
from pathlib import Path

from pydub import AudioSegment

logger = logging.getLogger(__name__)


def get_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def get_duration_minutes(path: Path) -> float:
    audio = AudioSegment.from_file(path)
    return len(audio) / (1000 * 60)


def compress_audio(input_path: Path, output_path: Path, bitrate: str) -> Path:
    logger.info("Compressing %s -> %s (bitrate=%s)", input_path, output_path, bitrate)
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="mp3", bitrate=bitrate)
    return output_path


def split_audio(
    input_path: Path, out_dir: Path, chunk_minutes: int, bitrate: str
) -> list[Path]:
    audio = AudioSegment.from_file(input_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunk_paths: list[Path] = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = out_dir / f"chunk_{i:03d}.mp3"
        chunk.export(chunk_path, format="mp3", bitrate=bitrate)
        chunk_paths.append(chunk_path)
        logger.info("Created chunk %s", chunk_path)
    return chunk_paths
