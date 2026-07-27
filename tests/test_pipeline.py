from __future__ import annotations

import math
from pathlib import Path

import pytest

from transcription.core import pipeline
from transcription.core.config import TranscriptionParams
from transcription.core.errors import ApiError, TranscriptionError

DIARIZED = TranscriptionParams(chunk_minutes=10, max_size_mb=24)
PLAIN = TranscriptionParams(model="whisper-1", response_format="json", chunk_minutes=10)
# The shipped defaults: chunk_minutes=45, capped to 23 by the model's own
# per-request duration limit.
DEFAULTS = TranscriptionParams()


class AudioStub:
    """Replaces the ffmpeg layer so the tests run instantly and offline.

    Everything it fakes is I/O — file sizes, durations, produced chunks — so
    what remains under test is the decision logic: convert or split, how many
    requests to make, and in what order results are handed back.
    """

    def __init__(self, duration_minutes, input_size_mb=5.0, transcoded_size_mb=5.0, chunks=3):
        self.duration_minutes = duration_minutes
        self.input_size_mb = input_size_mb
        self.transcoded_size_mb = transcoded_size_mb
        self.chunks = chunks
        self.transcode_calls = 0
        self.split_calls = 0
        self.chunk_seconds: float | None = None
        self.transcribed: list[str] = []
        self.fail_on_call: int | None = None

    def install(self, monkeypatch):
        monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: None)
        monkeypatch.setattr(
            pipeline, "probe_duration_minutes", lambda path: self.duration_minutes
        )
        monkeypatch.setattr(pipeline, "get_size_mb", self._size_mb)
        monkeypatch.setattr(pipeline, "transcode_to_mp3", self._transcode)
        monkeypatch.setattr(pipeline, "split_to_mp3_chunks", self._split)
        monkeypatch.setattr(pipeline, "transcribe_audio", self._transcribe)
        return self

    def _size_mb(self, path: Path) -> float:
        return self.transcoded_size_mb if path.name == "normalized.mp3" else self.input_size_mb

    def _transcode(self, input_path, output_path, bitrate):
        self.transcode_calls += 1
        return output_path

    def _split(self, input_path, out_dir, chunk_seconds, bitrate):
        self.split_calls += 1
        self.chunk_seconds = chunk_seconds
        return [out_dir / f"chunk_{i:03d}.mp3" for i in range(self.chunks)]

    def _transcribe(self, client, path, params):
        self.transcribed.append(path.name)
        if self.fail_on_call is not None and len(self.transcribed) == self.fail_on_call:
            raise ApiError("simulated API failure")
        return f"text of {path.stem}"


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "recording.m4a"
    path.write_bytes(b"pretend audio")
    return path


def test_missing_input_is_reported_before_any_work(source, monkeypatch):
    stub = AudioStub(duration_minutes=1).install(monkeypatch)
    with pytest.raises(TranscriptionError, match="not found"):
        pipeline.transcribe(source.parent / "gone.m4a", PLAIN, client=None)
    assert stub.transcribed == []


def test_short_small_audio_is_sent_as_one_request(source, monkeypatch):
    stub = AudioStub(duration_minutes=4, transcoded_size_mb=3).install(monkeypatch)

    result = pipeline.transcribe(source, PLAIN, client=None)

    assert (stub.transcode_calls, stub.split_calls) == (1, 0)
    assert result.was_split is False
    assert result.text == "text of normalized"


def test_audio_longer_than_chunk_minutes_is_split(source, monkeypatch):
    # Splitting here is not about size: long single requests are the ones
    # that get dropped by flaky connections before the response arrives.
    stub = AudioStub(duration_minutes=35, input_size_mb=8, chunks=4).install(monkeypatch)

    result = pipeline.transcribe(source, PLAIN, client=None)

    assert stub.split_calls == 1
    assert stub.transcode_calls == 0  # chunks are cut from the original file
    assert result.was_split is True
    assert len(result.chunks) == 4


def test_short_audio_still_over_the_limit_falls_back_to_splitting(source, monkeypatch):
    stub = AudioStub(
        duration_minutes=8, input_size_mb=60, transcoded_size_mb=40
    ).install(monkeypatch)

    result = pipeline.transcribe(source, PLAIN, client=None)

    assert (stub.transcode_calls, stub.split_calls) == (1, 1)
    assert result.was_split is True


def test_chunks_are_joined_with_the_format_separator(source, monkeypatch):
    AudioStub(duration_minutes=25, chunks=2).install(monkeypatch)

    plain = pipeline.transcribe(source, PLAIN, client=None)
    diarized = pipeline.transcribe(source, DIARIZED, client=None)

    assert plain.text == "text of chunk_000 text of chunk_001"
    assert diarized.text == "text of chunk_000\n\ntext of chunk_001"


def test_each_chunk_is_handed_over_as_soon_as_it_arrives(source, monkeypatch):
    """The point of on_partial: results already paid for survive a later failure."""
    stub = AudioStub(duration_minutes=25, chunks=3).install(monkeypatch)
    stub.fail_on_call = 3
    delivered: list[tuple[int, str]] = []

    with pytest.raises(ApiError):
        pipeline.transcribe(
            source, PLAIN, client=None, on_partial=lambda i, t: delivered.append((i, t))
        )

    # The two chunks that succeeded were delivered before the third failed.
    assert delivered == [(0, "text of chunk_000"), (1, "text of chunk_001")]


def test_progress_reports_the_chunk_count(source, monkeypatch):
    AudioStub(duration_minutes=25, chunks=3).install(monkeypatch)
    updates: list[pipeline.Progress] = []

    pipeline.transcribe(source, PLAIN, client=None, on_progress=updates.append)

    transcribing = [u for u in updates if u.stage == "transcribing"]
    assert [(u.current, u.total) for u in transcribing] == [(1, 3), (2, 3), (3, 3)]
    assert updates[-1].stage == "finished"


def test_a_broken_progress_callback_does_not_lose_the_transcript(source, monkeypatch):
    # The bot reports progress by calling the Telegram API, which can fail at
    # any moment; that must not throw away work that already cost money.
    AudioStub(duration_minutes=25, chunks=2).install(monkeypatch)

    def explode(progress):
        raise RuntimeError("status message could not be edited")

    result = pipeline.transcribe(source, PLAIN, client=None, on_progress=explode)

    assert len(result.chunks) == 2


# --- request size limits ----------------------------------------------------


def test_the_model_duration_limit_overrides_a_larger_chunk_minutes(source, monkeypatch):
    """gpt-4o-transcribe-diarize rejects anything over 1400 seconds.

    That limit belongs to the model, so asking for longer requests must be
    silently reduced to what the API accepts rather than produce an HTTP 400.
    """
    stub = AudioStub(duration_minutes=20, transcoded_size_mb=9).install(monkeypatch)

    # chunk_minutes=45, but the model caps a request at 23 minutes.
    result = pipeline.transcribe(source, DEFAULTS, client=None)

    assert result.was_split is False  # 20 minutes still fits in one request
    assert stub.split_calls == 0

    stub = AudioStub(duration_minutes=30, transcoded_size_mb=14).install(monkeypatch)
    result = pipeline.transcribe(source, DEFAULTS, client=None)

    # 30 minutes does not, even though it is well under chunk_minutes and
    # under max_size_mb.
    assert result.was_split is True


def test_a_recording_is_divided_into_equal_chunks(source, monkeypatch):
    # 29 minutes under a 23-minute limit needs two requests. Cutting at the
    # limit would leave a 6-minute tail, which is the least useful piece to
    # give a diarizer; two halves are better.
    stub = AudioStub(duration_minutes=29, chunks=2).install(monkeypatch)

    pipeline.transcribe(source, DEFAULTS, client=None)

    assert stub.chunk_seconds == pytest.approx(
        29 * 60 / 2 + pipeline.SPLIT_MARGIN_SECONDS
    )


def test_chunks_reach_past_the_end_of_the_recording(source, monkeypatch):
    """Otherwise ffmpeg opens one more file and writes nothing into it.

    A sub-second trailing chunk is rejected by the API as a corrupt file, so a
    boundary landing just short of the end fails a job whose real chunks all
    succeeded.
    """
    for duration in (29, 46.5, 47, 70.02, 200):
        stub = AudioStub(duration_minutes=duration).install(monkeypatch)
        pipeline.transcribe(source, DEFAULTS, client=None)

        chunk_count = math.ceil(duration * 60 / stub.chunk_seconds)
        assert stub.chunk_seconds * chunk_count > duration * 60


def test_chunks_never_exceed_the_model_limit(source, monkeypatch):
    for duration in (24, 47, 70, 200):
        stub = AudioStub(duration_minutes=duration).install(monkeypatch)
        pipeline.transcribe(source, DEFAULTS, client=None)
        assert stub.chunk_seconds is not None
        assert stub.chunk_seconds <= DEFAULTS.max_request_minutes * 60
