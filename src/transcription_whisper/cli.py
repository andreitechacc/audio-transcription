from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .audio import compress_audio, get_size_mb, split_audio
from .config import Config
from .transcriber import transcribe_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file via the OpenAI Whisper API"
    )
    parser.add_argument("input", type=Path, help="Path to the audio file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to save the transcript (default: <input>.txt)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="Language code, overrides config.toml (e.g. en, pl)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (default: ./config.toml)",
    )
    return parser.parse_args()


def run(input_path: Path, output_path: Path, language: str, config: Config) -> None:
    client = OpenAI()

    with tempfile.TemporaryDirectory(prefix="transcribe_") as tmp:
        tmp_dir = Path(tmp)
        size = get_size_mb(input_path)
        logger.info("Input size: %.1f MB", size)

        if size <= config.max_size_mb:
            text = transcribe_file(client, input_path, language, config.model)
        else:
            compressed_path = tmp_dir / "compressed.mp3"
            compress_audio(input_path, compressed_path, config.bitrate)
            compressed_size = get_size_mb(compressed_path)
            logger.info("Size after compression: %.1f MB", compressed_size)

            if compressed_size <= config.max_size_mb:
                text = transcribe_file(client, compressed_path, language, config.model)
            else:
                logger.info("Compression wasn't enough, splitting into chunks...")
                chunks = split_audio(
                    input_path, tmp_dir, config.chunk_minutes, config.bitrate
                )
                texts = [
                    transcribe_file(client, chunk, language, config.model)
                    for chunk in chunks
                ]
                text = " ".join(texts)

    output_path.write_text(text, encoding="utf-8")
    logger.info("Saved to %s", output_path)


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    config = Config.load(args.config)
    language = args.language or config.language
    output_path = args.output or args.input.with_suffix(".txt")

    run(args.input, output_path, language, config)


if __name__ == "__main__":
    main()
