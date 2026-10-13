"""Runtime invariant checks for the retrospective benchmarks.

Every comparison in this package rests on invariants that are asserted in prose
and enforced nowhere: that two arms were fitted on the same rows, that the
evaluation rows were never trained on, that a budget was spent exactly once,
that an "auxiliary" column is not the primary target's answer relabelled. None
of them crashes when violated. They move a number, quietly, and neither a
reader nor the author can tell from the output that they did.

``scripts/dense_block_campaign.py`` and ``scripts/dense_block_h1.py`` grew
ad-hoc versions of these checks (FINDINGS section 29); the instruments behind
sections 14-17 grew none. This is the same idea factored out, so those get the
identical treatment rather than a second hand-rolled one, and so the checks can
be unit-tested -- a check that has never fired is not evidence that it works.

Two design choices worth stating:

* **Collect, do not raise.** A violation in one cell of a grid should not
  destroy the other three hundred. Everything is accumulated and printed at the
  end with a count, so a run that trips a check still produces the table that
  shows *how much* the violation was worth.
* **Off by default, free when off.** Every entry point returns immediately when
  the audit is disabled, so leaving the calls in the hot loop costs nothing in
  the runs that do not ask for them.

Usage::

    a = Audit(enabled=args.audit)
    for ...:
        a.cell()
        a.exactly(tr, n, "primary training rows")
        a.disjoint(te, tr, "evaluation", "training")
        a.same_across(("JAK2", 20, 3), h, "te", te)
    a.report("[C]")
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

__all__ = ["Audit"]


def _key(idx) -> bytes:
    """A small, order-insensitive fingerprint of an index array.

    A digest rather than the sorted tuple it replaced: these are retained for
    the whole run, one per cell per tracked quantity, and on a large grid the
    tuples of boxed Python ints ran to hundreds of megabytes. 16 bytes each
    holds the same information for the only use there is -- equality.
    """
    a = np.sort(np.asarray(idx).ravel())
    return hashlib.blake2b(a.tobytes(), digest_size=16).digest()


class Audit:
    """Accumulates invariant violations for one instrument run.

    ``enabled`` false makes every method a no-op, which is why the call sites
    do not need to be guarded by ``if args.audit``.
    """

    def __init__(self, enabled: bool = False):
        self.on = bool(enabled)
        self.problems: list[str] = []
        self.checks = 0
        self.cells = 0
        # (group, name) -> {variant: fingerprint}. Used to verify that arms
        # which are differenced against each other actually shared a split.
        self._across: dict[tuple, dict] = {}

    # -- primitives --------------------------------------------------------

    def cell(self) -> None:
        """Count experimental cells, so ``report`` can say what was covered."""
        if self.on:
            self.cells += 1

    def want(self, cond, msg: str) -> bool:
        """Record ``msg`` unless ``cond`` holds. Returns whether it held."""
        if not self.on:
            return True
        self.checks += 1
        if not cond:
            self.problems.append(msg)
            return False
        return True

    # -- index-set invariants ---------------------------------------------

    def exactly(self, idx, size: int, what: str) -> None:
        """``idx`` holds exactly ``size`` rows and no row twice.

        The budget claim. An arm that quietly draws a row twice has spent one
        fewer observation than the column header says it did.
        """
        if not self.on:
            return
        idx = np.asarray(idx)
        self.want(idx.size == size,
                  f"{what}: {idx.size} rows, expected {size}")
        dup = idx.size - len(set(idx.ravel().tolist()))
        self.want(dup == 0, f"{what}: {dup} duplicate rows")

    def disjoint(self, a, b, a_name: str, b_name: str) -> None:
        """No row appears in both. The leakage check."""
        if not self.on:
            return
        shared = np.intersect1d(np.asarray(a), np.asarray(b))
        self.want(shared.size == 0,
                  f"{shared.size} rows are in both {a_name} and {b_name}")

    def subset(self, a, b, a_name: str, b_name: str) -> None:
        """Every row of ``a`` is in ``b``. The eligibility check."""
        if not self.on:
            return
        stray = np.setdiff1d(np.asarray(a), np.asarray(b))
        self.want(stray.size == 0,
                  f"{stray.size} rows of {a_name} fall outside {b_name}")

    # -- label invariants --------------------------------------------------

    def not_relabelled(self, primary, aux_columns, what: str) -> None:
        """No auxiliary column equals the primary's values.

        The failure this exists for is not a typo in an index; it is a column
        selected by name that turns out to *be* the target, at which point the
        multi-task arm is being handed the answer and every number in the table
        is a measurement of nothing.
        """
        if not self.on:
            return
        p = np.asarray(primary, dtype=float)
        for j, col in enumerate(aux_columns):
            c = np.asarray(col, dtype=float)
            if c.shape != p.shape:
                continue
            self.want(not np.allclose(c, p, equal_nan=True),
                      f"{what}: auxiliary column {j} equals the primary's values")

    def excludes(self, name, pool, what: str) -> None:
        """``name`` is not in ``pool`` -- a target used as its own auxiliary."""
        if not self.on:
            return
        self.want(name not in pool, f"{what}: {name} is its own auxiliary")

    # -- cross-arm invariants ---------------------------------------------

    def same_across(self, group, variant, name: str, idx) -> None:
        """Record ``idx`` so ``report`` can check it matched across variants.

        A gain is reported as ``rho(H) - rho(0)`` and read as a paired
        difference. That reading is only earned if the two arms were fitted on
        the same primary rows and scored on the same held-out rows. When they
        were not, the difference is still unbiased but carries the split-to-
        split variance twice over -- which is exactly the variance a negative
        result has to have removed before it can be called a null.
        """
        if not self.on:
            return
        self._across.setdefault((group, name), {})[variant] = _key(idx)

    def _across_problems(self) -> list[str]:
        """One line per quantity that failed to match across arms, not per cell.

        A split that varies with the arm varies in every cell, and 1,200 copies
        of the same sentence would bury the checks that failed once.
        """
        bad: Counter = Counter()
        total: Counter = Counter()
        for (_group, name), by_variant in self._across.items():
            total[name] += 1
            bad[name] += len(set(by_variant.values())) > 1
        return [f"{name}: differs across arms in {bad[name]} of {total[name]} "
                "cells -- the comparison is not paired"
                for name in sorted(total) if bad[name]]

    @property
    def all_problems(self) -> list[str]:
        """Direct violations and cross-arm mismatches, in one list."""
        return self.problems + self._across_problems()

    # -- output ------------------------------------------------------------

    def report(self, title: str = "", *, clean_lines=()) -> bool:
        """Print the verdict. Returns True when every invariant held."""
        if not self.on:
            return True
        problems = self.all_problems
        head = f"    [audit]{(' ' + title) if title else ''}"
        print(f"{head} {self.checks:,} checks over {self.cells:,} cells")
        if problems:
            print(f"    [audit] {len(problems)} PROBLEM(S):")
            for p in problems[:12]:
                print(f"      - {p}")
            if len(problems) > 12:
                print(f"      ... and {len(problems) - 12} more")
            return False
        for line in clean_lines:
            print(f"            {line}")
        return True

    def to_json(self) -> dict:
        """The audit result, for the run's JSON so it travels with the numbers."""
        problems = self.all_problems if self.on else []
        return {"enabled": self.on, "checks": self.checks, "cells": self.cells,
                "clean": bool(self.on and not problems), "problems": problems[:50]}
