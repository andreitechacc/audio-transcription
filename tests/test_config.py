from __future__ import annotations

import pytest

from transcription.core import config
from transcription.core.config import TranscriptionParams, load_params
from transcription.core.errors import ConfigError


def write_config(tmp_path, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_missing_file_falls_back_to_defaults(tmp_path):
    params = load_params(tmp_path / "does-not-exist.toml")
    assert params == TranscriptionParams()


def test_values_are_read_from_file(tmp_path):
    path = write_config(
        tmp_path,
        """
        language = "ru"
        model = "whisper-1"
        response_format = "json"
        chunk_minutes = 5
        """,
    )
    params = load_params(path)
    assert (params.language, params.model, params.chunk_minutes) == ("ru", "whisper-1", 5)
    # Keys the file did not mention keep their defaults.
    assert params.bitrate == TranscriptionParams.bitrate


def test_unknown_keys_are_ignored_with_a_warning(tmp_path, caplog):
    path = write_config(tmp_path, 'language = "ru"\nchunk_minuts = 5\n')
    params = load_params(path)
    assert params.chunk_minutes == TranscriptionParams.chunk_minutes
    assert "chunk_minuts" in caplog.text


def test_malformed_toml_reports_the_file(tmp_path):
    path = write_config(tmp_path, "language = ")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_params(path)


def test_diarized_format_requires_a_diarizing_model():
    # The API's own error for this combination is opaque, so it is rejected
    # before a request is ever sent.
    with pytest.raises(ConfigError, match="diarizing model"):
        TranscriptionParams(model="whisper-1", response_format="diarized_json")


def test_unknown_response_format_is_rejected():
    with pytest.raises(ConfigError, match="Unknown response_format"):
        TranscriptionParams(response_format="mp3")


@pytest.mark.parametrize("chunk_minutes", [0, -1])
def test_chunk_minutes_must_be_positive(chunk_minutes):
    with pytest.raises(ConfigError, match="chunk_minutes"):
        TranscriptionParams(chunk_minutes=chunk_minutes)


def test_chunk_separator_depends_on_format():
    # Diarized output is line-oriented; a plain transcript is flowing text.
    assert TranscriptionParams().chunk_separator == "\n\n"
    assert (
        TranscriptionParams(model="whisper-1", response_format="json").chunk_separator
        == " "
    )


def test_every_model_is_capped_by_default():
    """Each model the API offers limits how much audio one request may carry.

    A model absent from the override table must still be capped, otherwise the
    first recording longer than its unknown limit is rejected outright.
    """
    for model in ("gpt-4o-transcribe", "whisper-1", "some-future-model"):
        params = TranscriptionParams(
            model=model, response_format="json", chunk_minutes=120
        )
        assert params.max_request_minutes == config.DEFAULT_MAX_DURATION_MINUTES


def test_a_lower_chunk_minutes_still_wins():
    # The cap is an upper bound, not an override of an explicit preference.
    params = TranscriptionParams(
        model="gpt-4o-transcribe", response_format="json", chunk_minutes=5
    )
    assert params.max_request_minutes == 5


def test_a_confirmed_model_limit_overrides_the_default(monkeypatch):
    monkeypatch.setitem(config.MODEL_MAX_DURATION_MINUTES, "slow-model", 8.0)
    params = TranscriptionParams(
        model="slow-model", response_format="json", chunk_minutes=45
    )
    assert params.max_request_minutes == 8.0
