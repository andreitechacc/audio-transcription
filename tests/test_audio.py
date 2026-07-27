from __future__ import annotations

from pathlib import Path

from transcription.core import audio
from transcription.core.errors import AudioError


def test_empty_trailing_chunks_are_discarded(tmp_path, monkeypatch):
    """ffmpeg's segment muxer can leave a sliver the API rejects as corrupt.

    It ends a segment slightly after the requested time, so a boundary near
    the end of the recording still opens one more output file with nothing
    (or nothing decodable) in it.
    """
    good, sliver, broken = (tmp_path / f"chunk_{i:03d}.mp3" for i in range(3))
    for path in (good, sliver, broken):
        path.write_bytes(b"x")

    durations = {good: 868.0, sliver: 0.05}

    def fake_probe(path: Path) -> float:
        if path not in durations:
            raise AudioError("not decodable")  # ffprobe cannot read it at all
        return durations[path] / 60

    monkeypatch.setattr(audio, "probe_duration_minutes", fake_probe)

    assert audio._drop_unusable_chunks([good, sliver, broken]) == [good]
    # The rejects are removed rather than left in the working directory.
    assert not sliver.exists() and not broken.exists()
