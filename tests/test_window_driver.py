"""The window driver plans the same argv it always did, plus one new axis.

`--history-excludes-top` used to be hard-coded into every cell. FINDINGS 40
showed it is not a neutral control -- it removes the §22 confound (a history
that contains the answer) and introduces its mirror image (an arm that avoids
the history walks toward an answer the protocol guarantees is outside it) -- so
an experiment has to be able to drop it.

The risk in making a hard-coded flag optional is that an existing experiment
quietly replans without it, which would make 12 stored cells unreproducible
while still appearing to be "already present". These tests pin that down.

`clean_size` reads the real blocks, so each of these costs one block build. They
are kept to the two experiments that matter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_window_experiments as rwe  # noqa: E402

FLAG = "--history-excludes-top"


@pytest.fixture(scope="module")
def planned():
    """Every experiment's plan, built once."""
    return {name: rwe.plan(name, spec)
            for name, spec in rwe.EXPERIMENTS.items()}


def test_every_stored_experiment_still_plans_the_flag(planned):
    """The stored cells were run with it. If any of these loses the flag, the
    artefacts on disk stop matching the plan that claims to reproduce them."""
    for name in ("window", "power", "coverage", "acquisition", "oracle",
                 "controls"):
        for argv, path, label in planned[name]:
            assert FLAG in argv, f"{name}/{label} lost {FLAG}"


def test_the_leak_experiments_drop_it(planned):
    """And the new ones are the only ones that do."""
    for name in ("leak", "leakcov"):
        for argv, path, label in planned[name]:
            assert FLAG not in argv, f"{name}/{label} still carries {FLAG}"


def test_the_two_sides_are_otherwise_identical(planned):
    """`power` and `leak` must differ in the flag and the output path alone.

    Otherwise the comparison between them is not a controlled one, which is the
    entire point of running it.
    """
    by_label = {label: argv for argv, _, label in planned["power"]}
    assert by_label, "power planned nothing"
    for argv, path, label in planned["leak"]:
        assert label in by_label, f"leak cell {label} has no `power` twin"
        a = [x for x in by_label[label] if x != FLAG]
        b = list(argv)
        # The --out path is the one legitimate difference.
        i, j = a.index("--out"), b.index("--out")
        assert a[i + 1] != b[j + 1], "the two sides would overwrite each other"
        del a[i:i + 2]
        del b[j:j + 2]
        assert a == b, f"{label} differs by more than the flag:\n{a}\n{b}"


def test_leak_writes_to_its_own_prefix(planned):
    for _, path, _ in planned["leak"]:
        assert path.name.startswith("winleak_")
    for _, path, _ in planned["leakcov"]:
        assert path.name.startswith("leakcov_")


def test_default_is_to_keep_the_flag():
    """A spec that says nothing keeps the old behaviour, so a future experiment
    added without thinking about this does not silently change design."""
    assert all("excludes_top" not in spec
               for name, spec in rwe.EXPERIMENTS.items()
               if name not in ("leak", "leakcov")), (
        "an existing experiment grew an explicit excludes_top; check it matches "
        "what its stored artefacts were run with")
