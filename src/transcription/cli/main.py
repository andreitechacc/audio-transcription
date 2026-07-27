from __future__ import annotations

import argparse
import contextlib
import dataclasses
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv

from ..core.config import KNOWN_FORMATS, TranscriptionParams, load_params
from ..core.errors import TranscriptionError
from ..core.pipeline import Progress, transcribe
from ..core.transcriber import build_client

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe an audio file via the OpenAI speech-to-text API.",
    )
    parser.add_argument("input", type=Path, help="Path to the audio file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Where to save the transcript (default: alongside the input, as .txt)",
    )
    parser.add_argument(
        "-l", "--language", default=None, help="Language code, e.g. en, ru, pl"
    )
    parser.add_argument(
        "-m", "--model", default=None, help="Override the model from the config file"
    )
    parser.add_argument(
        "-f",
        "--response-format",
        default=None,
        choices=sorted(KNOWN_FORMATS),
        help="Override the response format from the config file",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (default: ./config.toml)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Also append logs to this file (console only by default)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Log the ffmpeg commands too"
    )
    return parser.parse_args(argv)


def setup_logging(verbose: bool, log_file: Path | None) -> None:
    """Configure logging once, at startup.

    Deliberately called from main() rather than at import time: importing a
    module should not reconfigure logging for whatever program imported it —
    which matters as soon as the bot and the web app import this same core.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )


def apply_overrides(
    params: TranscriptionParams, args: argparse.Namespace
) -> TranscriptionParams:
    """Layer CLI flags on top of the config file's values.

    dataclasses.replace re-runs the validation in __post_init__, so an
    invalid combination introduced by a flag (say, diarized output with a
    non-diarizing model) is caught here rather than by the API.
    """
    overrides = {
        field: value
        for field, value in (
            ("language", args.language),
            ("model", args.model),
            ("response_format", args.response_format),
        )
        if value is not None
    }
    return dataclasses.replace(params, **overrides) if overrides else params


def _make_progress_printer(stream: TextIO) -> Callable[[Progress], None]:
    """Print progress for a human, separately from the log stream."""

    def print_progress(progress: Progress) -> None:
        prefix = (
            f"[{progress.current}/{progress.total}] "
            if progress.total > 1
            else ""
        )
        print(f"{prefix}{progress.message}", file=stream, flush=True)

    return print_progress


def run(args: argparse.Namespace) -> None:
    params = apply_overrides(load_params(args.config), args)
    output_path = args.output or args.input.with_suffix(".txt")
    client = build_client()

    # Written chunk by chunk as results arrive, so a failure partway through
    # leaves the parts that already succeeded (and were already billed) on
    # disk instead of discarding them. The file is opened on the first chunk
    # rather than up front, so a run that fails before producing any text
    # (missing input, no ffmpeg, API refusal) leaves no empty file behind.
    with contextlib.ExitStack() as stack:
        out: TextIO | None = None

        def write_chunk(index: int, text: str) -> None:
            nonlocal out
            if out is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                out = stack.enter_context(open(output_path, "w", encoding="utf-8"))
            elif index > 0:
                out.write(params.chunk_separator)
            out.write(text)
            out.flush()

        transcribe(
            args.input,
            params,
            client,
            on_progress=_make_progress_printer(sys.stderr),
            on_partial=write_chunk,
        )

    print(f"Saved transcript to {output_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    setup_logging(args.verbose, args.log_file)

    try:
        run(args)
    except TranscriptionError as exc:
        # Everything the user can actually fix arrives as this type, so it is
        # shown as a message. Anything else is a bug and keeps its traceback.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
