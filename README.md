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
chunk_minutes = 10      # split into chunks this long even if under max_size_mb
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
temporary directory; it's deleted automatically on success, but **kept on
failure** so a failed run can be inspected or retried without recompressing.

Every run also appends to `transcribe.log` (timestamps, sizes, per-request
OpenAI request IDs, errors) — useful if a request fails or if you need to
reference a specific request when contacting OpenAI support.

## How the size limit is handled

1. If the file is shorter than `chunk_minutes`, it's re-encoded to MP3 at
   `bitrate` and sent as one request. Re-encoding isn't just about size —
   some phone recordings are saved as `.m4a` with a container variant that
   OpenAI's format check rejects even though m4a is nominally supported, and
   re-encoding sidesteps that. (If it's still over `max_size_mb` after that,
   it falls through to splitting.)
2. Otherwise, it's split directly from the original file into
   `chunk_minutes`-long MP3 chunks (each chunk is re-encoded once as part of
   the split, so there's no separate whole-file compression step first).
   Splitting also happens even when the file is under the size limit, since
   a single long request is more likely to have its connection dropped
   before the response comes back than several short ones. Each chunk's
   text is written to the output file
   immediately after it comes back, so if a later chunk fails, the earlier
   (already paid-for) results aren't lost.

## Network reliability

Requests use a 120s timeout and up to 5 automatic retries, since a single
transcription can legitimately take a minute or more and flaky connections
(VPNs, proxies) can otherwise drop it prematurely. If you're outside an
[OpenAI-supported region](https://platform.openai.com/docs/supported-countries)
you'll need a VPN, but note that some VPN configurations disconnect long-lived
connections themselves — if requests keep failing, try a different VPN
server/protocol (WireGuard tends to hold up better than OpenVPN for this).

Note: if a request fails *after* OpenAI already processed the audio (e.g. the
response was lost in transit), you may still be billed for it — the API has
no way to re-fetch a lost transcription result after the fact, only the
request ID in `transcribe.log` for reference.

## License

MIT — see [LICENSE](LICENSE).
