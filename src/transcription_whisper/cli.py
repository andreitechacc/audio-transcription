from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .audio import compress_audio, get_size_mb, split_audio
from .config import Config
from .transcriber import transcribe_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("transcribe.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Give the client more patience than the default (a few seconds) and more
# automatic retries, since a single transcription request can legitimately
# take a minute or more and flaky connections (VPNs, proxies) can drop it.
REQUEST_TIMEOUT_SECONDS = 120.0
MAX_RETRIES = 5


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
    client = OpenAI(timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES)

    tmp_dir = Path(tempfile.mkdtemp(prefix="transcribe_"))
    try:
        size = get_size_mb(input_path)
        logger.info("Input size: %.1f MB", size)

        # Always re-encode through ffmpeg first, even for small files: some
        # phone recordings are saved as .m4a with a container brand (e.g.
        # 3gp4) that OpenAI's format sniffing rejects, even though m4a is
        # nominally supported. Re-encoding to a plain mp3 sidesteps that.
        normalized_path = tmp_dir / "normalized.mp3"
        compress_audio(input_path, normalized_path, config.bitrate)
        normalized_size = get_size_mb(normalized_path)
        logger.info("Normalized size: %.1f MB", normalized_size)

        if normalized_size <= config.max_size_mb:
            text = transcribe_file(client, normalized_path, language, config.model)
            output_path.write_text(text, encoding="utf-8")
        else:
            logger.info("Still over the limit after re-encoding, splitting into chunks...")
            chunks = split_audio(
                normalized_path, tmp_dir, config.chunk_minutes, config.bitrate
            )
            # Write each chunk's text as soon as it comes back, so a
            # failure partway through doesn't lose the chunks that
            # already succeeded (and were already paid for).
            with open(output_path, "w", encoding="utf-8") as out:
                for i, chunk in enumerate(chunks):
                    text = transcribe_file(client, chunk, language, config.model)
                    if i > 0:
                        out.write(" ")
                    out.write(text)
                    out.flush()
                    logger.info(
                        "Progress saved: chunk %d/%d written to %s",
                        i + 1,
                        len(chunks),
                        output_path,
                    )
    except Exception:
        logger.error(
            "Transcription failed; keeping temporary files for inspection/retry at %s",
            tmp_dir,
        )
        raise
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
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
