# transcription-whisper

A small CLI tool for transcribing audio files with the [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text).

Handles the API's 25 MB file size limit automatically: if the input file is too
large, it first tries re-encoding at a lower bitrate, and if that's still not
enough, it splits the audio into chunks and transcribes each one separately.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/) available on `PATH` (required by `pydub` for
  reading/encoding audio)
- An OpenAI API key

## Setup

```bash
git clone <your-repo-url>
cd transcription-whisper
uv sync
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

## Configuration

Non-secret defaults live in [`config.toml`](config.toml):

```toml
language = "en"        # default language code (e.g. "en", "pl")
model = "whisper-1"
max_size_mb = 24        # API limit is 25 MB, keep a small margin
bitrate = "64k"         # bitrate used when compressing/splitting
chunk_minutes = 10      # chunk length when splitting is needed
```

The `--language` and `--config` CLI flags override these per run.

## Usage

```bash
uv run transcribe path/to/audio.m4a
uv run transcribe path/to/audio.m4a --language pl
uv run transcribe path/to/audio.m4a -o result.txt
```

The transcript is written to `<input>.txt` by default (or the path passed to
`-o`/`--output`). Intermediate compressed/split files are created in a
temporary directory and cleaned up automatically.

## How the size limit is handled

1. If the file is under `max_size_mb`, it's sent as-is.
2. Otherwise, it's re-encoded to MP3 at `bitrate`. If that's small enough, it's
   sent.
3. Otherwise, the file is split into `chunk_minutes`-long MP3 chunks, each
   transcribed separately, and the resulting texts are joined with spaces.

## License

MIT — see [LICENSE](LICENSE).
