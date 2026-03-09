from __future__ import annotations

from pathlib import Path

from reguq.colab import bootstrap_colab_environment


def test_bootstrap_returns_false_when_marker_exists(tmp_path: Path):
    marker = tmp_path / "ready.marker"
    marker.write_text("ok", encoding="utf-8")

    changed = bootstrap_colab_environment(
        repo_url="https://github.com/DaneshSelwal/reguq.git",
        marker_path=str(marker),
    )

    assert changed is False


def test_bootstrap_installs_and_restarts_when_colab(monkeypatch, tmp_path: Path):
    marker = tmp_path / "ready.marker"

    calls = []

    def _fake_run(cmd, check):
        calls.append(cmd)

    kill_calls = []

    def _fake_kill(pid, sig):
        kill_calls.append((pid, sig))

    monkeypatch.setattr("reguq.colab.subprocess.run", _fake_run)
    monkeypatch.setattr("reguq.colab.is_colab_runtime", lambda: True)
    monkeypatch.setattr("reguq.colab.os.kill", _fake_kill)

    changed = bootstrap_colab_environment(
        repo_url="https://github.com/DaneshSelwal/reguq.git",
        marker_path=str(marker),
        quiet=True,
    )

    assert changed is True
    assert marker.exists()
    assert len(calls) == 3
    assert kill_calls
