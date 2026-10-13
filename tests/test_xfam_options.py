"""The 2x3 sweep must actually be a 2x3 sweep.

`run_xfam_options.py` measures three uses of a free history crossed with two
history designs, and the whole point is that the six cells differ in exactly
one thing each. A plan where two conditions share an output path, or where the
"withheld" arm quietly keeps a uniform history, would produce a comparison
table that looks fine and means nothing.

It also has to carry `--audit`. The first version of this sweep was run from a
shell loop without it, and the artefacts came back with `enabled: false` --
which reads as `clean: false` and is indistinguishable at a glance from an
audit that ran and failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_xfam_options as rxo  # noqa: E402

FLAG = "--history-excludes-top"


def test_every_cell_is_audited():
    for argv, _, label in rxo.plan(18):
        assert "--audit" in argv, f"{label} would produce an unaudited artefact"


def test_the_two_designs_differ_by_the_flag_alone():
    """Pair each uniform cell with its withheld twin and diff the argv."""
    by_label = {label: argv for argv, _, label in rxo.plan(18)}
    for name in ("xfam_transfer", "xfam_seed"):
        for i in range(1, 19):
            a = list(by_label[f"{name}/k{i:02d}"])
            b = list(by_label[f"{name}_excl/k{i:02d}"])
            assert FLAG not in a and FLAG in b
            b.remove(FLAG)
            ia, ib = a.index("--out"), b.index("--out")
            assert a[ia + 1] != b[ib + 1], "the two designs would collide"
            del a[ia:ia + 2]
            del b[ib:ib + 2]
            assert a == b, f"{name}/k{i:02d} differs by more than the flag"


def test_the_two_options_differ_by_the_seeding_alone():
    by_label = {label: argv for argv, _, label in rxo.plan(18)}
    for suffix in ("", "_excl"):
        a = list(by_label[f"xfam_transfer{suffix}/k01"])
        b = list(by_label[f"xfam_seed{suffix}/k01"])
        # transfer adds the `mt` arm; seed adds the seeding rule. Both keep
        # `stl`, which is the `abstain` option and the shared baseline.
        assert "mt" in a and "mt" not in b
        assert "--seed-pool" in b and "--seed-pool" not in a
        assert "stl" in a and "stl" in b


def test_every_cell_has_its_own_output():
    paths = [path for _, path, _ in rxo.plan(18)]
    assert len(set(paths)) == len(paths) == 72


def test_the_budget_is_inside_the_operating_window():
    """The blocks are 428-433 compounds and the window is 22-33% of the
    library. A budget outside it would measure a regime the rest of the
    package has already shown does nothing."""
    argv = rxo.plan(1)[0][0]
    budget = int(argv[argv.index("--budget") + 1])
    assert 0.20 <= budget / 430 <= 0.35


def test_kiba_keeps_its_censored_values():
    """KIBA runs with --keep-censored throughout: the filter leaves 25-99 of
    ~900 rows there, so applying it would measure a correlation no campaign
    sees. Dropping this flag would silently change which library is searched."""
    for argv, _, label in rxo.plan(2):
        assert "--keep-censored" in argv, label
