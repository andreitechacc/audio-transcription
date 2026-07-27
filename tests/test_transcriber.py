from __future__ import annotations

from types import SimpleNamespace

import pytest

from transcription.core.config import TranscriptionParams
from transcription.core.transcriber import transcribe_audio

DIARIZED = TranscriptionParams()
PLAIN = TranscriptionParams(model="whisper-1", response_format="json")


class FakeClient:
    """Stands in for the OpenAI client.

    Only the one call the code makes is implemented, and the request keyword
    arguments are recorded so tests can assert on what was actually sent.
    """

    def __init__(self, parsed):
        self.parsed = parsed
        self.calls: list[dict] = []
        create = self._create
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=create)
            )
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
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
