from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from transcription.core import transcriber
from transcription.core.config import TranscriptionParams
from transcription.core.errors import ApiError
from transcription.core.transcriber import transcribe_audio

DIARIZED = TranscriptionParams()
PLAIN = TranscriptionParams(model="whisper-1", response_format="json")


class FakeClient:
    """Stands in for the OpenAI client.

    Only the one call the code makes is implemented, and the request keyword
    arguments are recorded so tests can assert on what was actually sent.
    """

    def __init__(self, parsed=None, errors=()):
        self.parsed = parsed
        # Raised one per call, in order, before any successful response.
        self.errors = list(errors)
        self.calls: list[dict] = []
        create = self._create
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=create)
            )
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        parsed = self.parsed
        return SimpleNamespace(
            headers={"x-request-id": "req_test"}, parse=lambda: parsed
        )


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "clip.mp3"
    path.write_bytes(b"not really audio")
    return path


def test_diarized_response_becomes_speaker_labelled_lines(audio_file):
    client = FakeClient(
        SimpleNamespace(
            segments=[
                SimpleNamespace(speaker="A", text="  Hello there  "),
                SimpleNamespace(speaker="B", text="Hi."),
            ]
        )
    )
    assert transcribe_audio(client, audio_file, DIARIZED) == "[A] Hello there\n[B] Hi."


def test_plain_json_response_returns_its_text(audio_file):
    client = FakeClient(SimpleNamespace(text="just the words"))
    assert transcribe_audio(client, audio_file, PLAIN) == "just the words"


def test_raw_body_formats_are_passed_through(audio_file):
    # text/srt/vtt come back as a plain string rather than an object.
    client = FakeClient("1\n00:00:00,000 --> 00:00:01,000\nHello\n")
    params = TranscriptionParams(model="whisper-1", response_format="srt")
    assert transcribe_audio(client, audio_file, params).startswith("1\n")


def test_diarization_asks_the_api_to_segment_long_audio(audio_file):
    client = FakeClient(SimpleNamespace(segments=[]))
    transcribe_audio(client, audio_file, DIARIZED)
    # Without this the API diarizes each part separately and speaker labels
    # drift within a single request.
    assert client.calls[0]["chunking_strategy"] == "auto"


def test_plain_transcription_does_not_send_a_chunking_strategy(audio_file):
    client = FakeClient(SimpleNamespace(text=""))
    transcribe_audio(client, audio_file, PLAIN)
    assert "chunking_strategy" not in client.calls[0]


# --- retry policy -----------------------------------------------------------
#
# The distinction these tests protect is a financial one: a failure that
# happened before the server did any work is free to retry, while a timeout or
# a dropped connection means the audio was very likely transcribed and billed
# already, so retrying pays for the same work twice.

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")


def status_error(code: int) -> openai.APIStatusError:
    return openai.APIStatusError(
        "server said no", response=httpx.Response(code, request=REQUEST), body=None
    )


def rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError(
        "slow down", response=httpx.Response(429, request=REQUEST), body=None
    )


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Keep backoff from making the suite slow."""
    monkeypatch.setattr(transcriber.time, "sleep", lambda seconds: None)


def test_a_timeout_is_not_retried(audio_file):
    # Retrying would upload and pay for the same audio again.
    client = FakeClient(errors=[openai.APITimeoutError(request=REQUEST)])

    with pytest.raises(ApiError) as excinfo:
        transcribe_audio(client, audio_file, PLAIN)

    assert len(client.calls) == 1
    assert "billed" in str(excinfo.value)


def test_a_timeout_is_not_reported_as_a_connection_problem(audio_file):
    # APITimeoutError subclasses APIConnectionError, so an inattentive except
    # order sends the user off investigating their network for what is really
    # a too-short timeout.
    client = FakeClient(errors=[openai.APITimeoutError(request=REQUEST)])

    with pytest.raises(ApiError, match="timeout"):
        transcribe_audio(client, audio_file, PLAIN)


def test_a_dropped_connection_is_not_retried(audio_file):
    client = FakeClient(
        errors=[openai.APIConnectionError(message="connection lost", request=REQUEST)]
    )

    with pytest.raises(ApiError):
        transcribe_audio(client, audio_file, PLAIN)

    assert len(client.calls) == 1


def test_a_server_error_is_retried_and_can_succeed(audio_file):
    # The server refused before transcribing anything, so this retry is free.
    client = FakeClient(
        parsed=SimpleNamespace(text="recovered"), errors=[status_error(503)]
    )

    assert transcribe_audio(client, audio_file, PLAIN) == "recovered"
    assert len(client.calls) == 2


def test_a_rate_limit_is_retried(audio_file):
    client = FakeClient(parsed=SimpleNamespace(text="ok"), errors=[rate_limit_error()])

    assert transcribe_audio(client, audio_file, PLAIN) == "ok"
    assert len(client.calls) == 2


def test_retries_are_bounded(audio_file):
    client = FakeClient(errors=[status_error(500) for _ in range(10)])

    with pytest.raises(ApiError, match="kept failing"):
        transcribe_audio(client, audio_file, PLAIN)

    assert len(client.calls) == transcriber.MAX_SAFE_RETRIES + 1


def test_client_errors_are_not_retried(audio_file):
    # A 400 means the request itself is wrong; sending it again cannot help.
    client = FakeClient(errors=[status_error(400)])

    with pytest.raises(ApiError, match="HTTP 400"):
        transcribe_audio(client, audio_file, PLAIN)

    assert len(client.calls) == 1
