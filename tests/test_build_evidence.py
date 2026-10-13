"""The package must not be shippable while a check fails.

The advisor's objection was that the manuscript asserted checks pass without
shipping anything that would let a reader confirm it. The answer is a package
whose outputs are produced at build time -- so the one property worth pinning
is that a failing check cannot be papered over by a successful build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_evidence as be  # noqa: E402


def test_a_failing_check_fails_the_build(tmp_path, monkeypatch):
    """A build that reports success while a check exits nonzero would be
    worse than no package at all."""
    monkeypatch.setattr(be, "CHECKS", (
        ("passes", ["-c", "print('fine')"]),
        ("fails", ["-c", "import sys; sys.exit(3)"]),
    ))
    monkeypatch.setattr(be, "ARTEFACT_GLOBS", ())
    rc = be.main(["--out", str(tmp_path / "pkg")])
    assert rc == 1

    status = json.loads(
        (tmp_path / "pkg" / "outputs" / "status.json").read_text(
            encoding="utf-8"))
    assert status["all_passed"] is False
    assert status["checks"]["fails"] == 3


def test_the_exit_status_of_every_check_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(be, "CHECKS", (("ok", ["-c", "print('x')"]),))
    monkeypatch.setattr(be, "ARTEFACT_GLOBS", ())
    assert be.main(["--out", str(tmp_path / "pkg")]) == 0
    text = (tmp_path / "pkg" / "outputs" / "ok.txt").read_text(
        encoding="utf-8")
    assert "# exit status: 0" in text
    assert "x" in text


def test_the_manifest_covers_the_outputs_it_just_wrote(tmp_path, monkeypatch):
    """A manifest written before the checks ran would not cover their output,
    which is the part a reviewer most needs to tie to the files."""
    monkeypatch.setattr(be, "CHECKS", (("ok", ["-c", "print('x')"]),))
    monkeypatch.setattr(be, "ARTEFACT_GLOBS", ())
    be.main(["--out", str(tmp_path / "pkg")])
    manifest = (tmp_path / "pkg" / "MANIFEST.sha256").read_text(
        encoding="utf-8")
    assert "outputs/ok.txt" in manifest
    assert "README.md" in manifest
    assert "MANIFEST.sha256" not in manifest


def test_the_manifest_digests_are_the_files_digests(tmp_path, monkeypatch):
    monkeypatch.setattr(be, "CHECKS", ())
    monkeypatch.setattr(be, "ARTEFACT_GLOBS", ())
    pkg = tmp_path / "pkg"
    be.main(["--out", str(pkg)])
    for line in (pkg / "MANIFEST.sha256").read_text(
            encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        assert be.sha256(pkg / rel) == digest


def test_rebuilding_replaces_a_stale_package(tmp_path, monkeypatch):
    """A leftover file from a previous build would be shipped unexplained."""
    monkeypatch.setattr(be, "CHECKS", ())
    monkeypatch.setattr(be, "ARTEFACT_GLOBS", ())
    pkg = tmp_path / "pkg"
    be.main(["--out", str(pkg)])
    stale = pkg / "outputs" / "leftover.txt"
    stale.write_text("from an older build", encoding="utf-8")
    be.main(["--out", str(pkg)])
    assert not stale.exists()


@pytest.mark.skipif(not (ROOT / "results").exists(),
                    reason="results tree not present")
def test_the_real_check_list_names_scripts_that_exist():
    for name, cmd in be.CHECKS:
        script = ROOT / cmd[0]
        assert script.exists(), f"{name} points at a missing script: {cmd[0]}"
