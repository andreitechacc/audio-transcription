from __future__ import annotations

import pytest

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
