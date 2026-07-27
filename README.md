# transcription

[![CI](https://github.com/USERNAME/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/REPO/actions/workflows/ci.yml)

Transcribes audio files with the [OpenAI speech-to-text API](https://platform.openai.com/docs/guides/speech-to-text),
including speaker-labelled output. Long or oversized recordings are converted
and split automatically, so a two-hour meeting is handled the same way as a
two-minute voice note.

Currently a CLI; a Telegram bot and a web interface are planned on top of the
same core (see [Roadmap](#roadmap)).

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/) (both `ffmpeg` and `ffprobe`) on `PATH`
- An OpenAI API key

## Setup

```bash
git clone <your-repo-url>
cd transcription
uv sync
cp .env.example .env      # then set OPENAI_API_KEY in .env
```

## Usage

```bash
uv run transcribe data/meeting.m4a
uv run transcribe data/meeting.m4a --language ru
uv run transcribe data/meeting.m4a -o data/meeting-transcript.txt
uv run transcribe data/interview.m4a --model whisper-1 --response-format json
```

The transcript is written next to the input as `.txt` unless `-o` says
otherwise. Progress is printed as it goes:

```
Inspecting audio…
Splitting audio…
[1/3] Transcribing part 1 of 3…
[2/3] Transcribing part 2 of 3…
[3/3] Transcribing part 3 of 3…
Saved transcript to data/meeting.txt
```

The tool reads and writes wherever you point it. `data/` is only a
convention — it is git-ignored, so personal recordings and transcripts stay
out of the repository by default.

Logs go to the console. Pass `--log-file data/transcribe.log` to keep a copy
with timestamps, sizes and per-request OpenAI request IDs — useful when a
request fails and you need to reference it. `-v` additionally logs the exact
ffmpeg commands.

## Configuration

Non-secret defaults live in [`config.toml`](config.toml); CLI flags override
them per run.

```toml
language = "en"
model = "gpt-4o-transcribe-diarize"
response_format = "diarized_json"  # speaker-labelled; "json" for plain text
max_size_mb = 24                   # API limit is 25 MB, keep a margin
bitrate = "64k"
chunk_minutes = 10
```

`diarized_json` labels each line with a speaker (`[A] …`, `[B] …`) and
requires a diarizing model. That pairing is validated at startup, because the
API's own error for the wrong combination is hard to interpret.

## Architecture

```
src/transcription/
  core/      transcription logic — knows nothing about how it is invoked
  cli/       argument parsing, file output, console progress
```

`core` exposes one function, `transcribe(...)`, which **returns** the
transcript and reports progress through callbacks, rather than writing files
or printing anything itself. That split is deliberate: the CLI writes to disk
and prints to the console, the planned Telegram bot will edit a chat message
and send a document, and the web API will write rows to a database — all
calling the same function, with no duplicated transcription logic.

The `on_partial` callback delivers each chunk's text the moment it arrives
instead of at the end. Chunks are billed as they are transcribed, so a
failure on chunk 7 of 8 must not discard the seven that already cost money.

All audio work is done by invoking `ffmpeg`/`ffprobe` directly. Duration
comes from `ffprobe` reading container metadata — milliseconds, no decoding —
and splitting is a single `ffmpeg -f segment` pass. Nothing decodes audio
inside Python, so memory use stays flat regardless of recording length.

## How long and oversized audio is handled

1. **Shorter than `chunk_minutes`:** converted to a single mono MP3 and sent
   as one request. The conversion is not only about size — some phone
   recordings are `.m4a` with a container brand (e.g. `3gp4`) that the API
   rejects even though m4a is nominally supported, and a plain MP3 sidesteps
   that. If the result still exceeds `max_size_mb`, it falls through to
   splitting.
2. **Longer than `chunk_minutes`:** split directly from the original file
   into chunks, each transcribed separately. This happens even when the file
   is under the size limit: a single request covering a long recording takes
   long enough that flaky links (proxies, some VPNs) drop the connection
   before the response arrives, whereas several short requests each complete
   comfortably.

When splitting diarized audio, each chunk is diarized independently — speaker
labels are **not** guaranteed to refer to the same person across chunk
boundaries, since every request assigns them from scratch.

## Network reliability

Requests use a 120-second timeout and up to 5 automatic retries, well above
the SDK defaults, which are tuned for fast chat calls. If you are outside an
[OpenAI-supported region](https://platform.openai.com/docs/supported-countries)
you will need a VPN — but note that some VPN configurations drop long-lived
connections themselves. If requests keep failing, try a different server or
protocol (WireGuard holds up better than OpenVPN here).

If a request fails *after* OpenAI processed the audio (the response was lost
in transit), you may still be billed for it. There is no way to re-fetch a
lost result; only the request ID in the log remains for reference.

## Development

```bash
uv run pytest
```

The tests stub out ffmpeg and the API client, so they need neither installed
nor an API key, and they cost nothing to run. What they actually cover is the
decision logic — convert or split, how many requests, in what order results
are handed back — which is where a silent mistake would cost real money.

## Roadmap

- [x] CLI
- [ ] Telegram bot — send a voice message, get a transcript back
- [ ] Web interface — upload a file, track the job, download the result

## License

MIT — see [LICENSE](LICENSE).
