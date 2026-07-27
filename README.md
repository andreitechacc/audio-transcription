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
bitrate = "64k"                    # mono; 24 MB is reached at roughly 50 min
chunk_minutes = 45                 # upper bound on one request, not a target
request_timeout_seconds = 900
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

**Splitting is avoided wherever possible, because it costs quality.** The API
keeps speaker labels consistent within a single request, but assigns them from
scratch in the next one, so every split is a seam where `A` may become `B`.
The aim is therefore to send as much audio per request as the limits allow.

How much audio one request may carry is the stricter of two limits: the
`chunk_minutes` you configure, and the model's own hard cap —
`gpt-4o-transcribe-diarize` rejects anything over 1400 seconds (23 minutes)
outright. The model's limit is kept in code rather than in `config.toml`,
since raising it would only guarantee an HTTP 400.

1. **Within that limit:** converted to a single mono MP3 and sent as one
   request. The conversion is not only about size — some phone recordings are
   `.m4a` with a container brand (e.g. `3gp4`) that the API rejects even
   though m4a is nominally supported, and a plain MP3 sidesteps that. If the
   result still exceeds `max_size_mb`, it falls through to splitting.
2. **Beyond it:** split from the original file, each chunk transcribed
   separately.

When splitting, the recording is divided into **equal** parts rather than cut
at the limit. A 29-minute recording under a 23-minute cap becomes 14.5 + 14.5,
not 23 + 6: a short trailing fragment is the piece a diarizer has least
context for, so leaving one costs quality for no benefit.

## Network reliability

Transcription is billed per minute of audio as soon as the API processes it,
whether or not the response reaches you. That single fact shapes the retry
policy here, and it is why the SDK's own retries are switched off:

- **Timeouts and dropped connections are never retried automatically.** By
  then the audio has been uploaded and almost certainly transcribed and
  billed; only the response was lost. Retrying pays for the same work again
  without improving the odds. The failure is reported instead, so the decision
  to pay again is yours.
- **Rate limits and 5xx responses are retried** (twice, with backoff). The
  server refused before doing any work, so those retries are free.

`request_timeout_seconds` defaults to 900 because one request covering 45
minutes of audio can take several minutes to come back, during which the
connection carries nothing. Setting it too low is actively expensive: it
aborts requests the API has already transcribed and billed.

If you are outside an
[OpenAI-supported region](https://platform.openai.com/docs/supported-countries)
you will need a VPN — but some VPN configurations close connections that stay
idle, which is exactly what a long transcription looks like. A symptom worth
recognising: attempts failing after a suspiciously round interval (60 seconds,
say) point at the network path, whereas attempts failing at exactly
`request_timeout_seconds` point at this setting.

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
