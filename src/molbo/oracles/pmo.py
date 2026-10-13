"""Secondary tier: PMO tasks, for comparability with published baselines.

The primary tier is an oracle we built ourselves, which is exactly the thing
reviewers distrust. PMO (Gao et al., NeurIPS 2022) exists because so many
papers report on trivial or self-designed tasks; it evaluated 25 algorithms on
23 tasks under a controlled budget and found most "state-of-the-art" methods
fail to beat their predecessors once the budget is fixed. Running a subset of
it buys 25 baselines and removes the "you invented your own benchmark"
objection. Recent work restricts the budget further to 1,000 calls, which is
much closer to the regime this project cares about.

Two things have to match PMO exactly or the numbers are not comparable:

* **Budget is counted in unique canonical SMILES.** Already enforced by
  :class:`~molbo.oracles.base.Oracle`; PMO's own scorer memoises the same way.
* **Invalid molecules score 0.0 and still cost budget.** Also the base class's
  default. A method must not be able to buy extra evaluations by emitting
  garbage.

The scores are already in ``[0, 1]`` and higher-is-better, so unlike the ChEMBL
tier they need no rescaling before going into the AUC.

Compatibility note
------------------
TDC's oracle module still does ``from rdkit.six import iteritems``. RDKit
dropped its vendored ``six`` years ago, and TDC swallows the resulting
ImportError behind a misleading "Please install rdkit" message. The single use
is ``iteritems(fps)`` inside the synthetic-accessibility score, which is
``dict.items()`` on Python 3, so :func:`install_rdkit_six_shim` supplies
exactly that and nothing else. No oracle arithmetic is touched.
"""

from __future__ import annotations

import sys
import types
from typing import Sequence

from .base import Oracle

PMO_TASKS: tuple[str, ...] = (
    "albuterol_similarity",
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "deco_hop",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "jnk3",
    "median1",
    "median2",
    "mestranol_similarity",
    "osimertinib_mpo",
    "perindopril_mpo",
    "qed",
    "ranolazine_mpo",
    "scaffold_hop",
    "sitagliptin_mpo",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
    "valsartan_smarts",
    "zaleplon_mpo",
)
"""The 23 PMO tasks, in PMO's own naming."""

# TDC's Oracle() name lookup is fuzzy but inconsistent about spaces vs
# underscores; pin the spellings that actually resolve.
_TDC_NAME = {
    "median1": "Median 1",
    "median2": "Median 2",
    "deco_hop": "Deco Hop",
    "scaffold_hop": "Scaffold Hop",
    "isomers_c7h8n2o2": "Isomers_C7H8N2O2",
    "isomers_c9h10n2o2pf2cl": "Isomers_C9H10N2O2PF2Cl",
}

RDKIT_ONLY = frozenset(PMO_TASKS) - {"drd2", "gsk3b", "jnk3"}
"""Tasks that are pure RDKit. The other three load pickled scikit-learn models
whose compatibility depends on the installed sklearn version."""


def install_rdkit_six_shim() -> bool:
    """Register a minimal ``rdkit.six`` so TDC's oracle module imports.

    Returns True if the shim was installed, False if it was already importable.
    """
    try:
        import rdkit.six  # noqa: F401

        return False
    except ImportError:
        pass

    import rdkit

    mod = types.ModuleType("rdkit.six")
    mod.iteritems = lambda d, **kw: iter(d.items(**kw))
    mod.itervalues = lambda d, **kw: iter(d.values(**kw))
    mod.iterkeys = lambda d, **kw: iter(d.keys(**kw))
    mod.string_types = (str,)
    sys.modules["rdkit.six"] = mod
    rdkit.six = mod
    return True


class PMOOracle(Oracle):
    """One PMO task, wrapped in this project's budget accounting.

    Parameters
    ----------
    task:
        A name from :data:`PMO_TASKS`.
    budget:
        Unique oracle calls. 1,000 is the restricted setting used here; PMO's
        original tables use 10,000. Always report which -- they are not
        interchangeable.
    """

    def __init__(self, task: str, *, budget: int | None = 1000, **kwargs):
        task = task.lower().strip()
        if task not in PMO_TASKS:
            raise ValueError(
                f"unknown PMO task {task!r}; expected one of {sorted(PMO_TASKS)}"
            )
        install_rdkit_six_shim()
        try:
            from tdc import Oracle as TDCOracle
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the PMO tier needs PyTDC: uv pip install --no-deps PyTDC==0.4.17, "
                "plus fuzzywuzzy, seaborn, 'setuptools<81', mygene, huggingface_hub"
            ) from exc

        self.task = task
        self._tdc = TDCOracle(name=_TDC_NAME.get(task, task))

        # PMO's conventions, not ours: score 0.0 for anything unparseable, and
        # charge budget for it.
        kwargs.setdefault("invalid_value", 0.0)
        kwargs.setdefault("count_invalid", True)
        kwargs.setdefault("maximize", True)
        super().__init__(f"pmo:{task}", budget=budget, **kwargs)

    @property
    def cache_key(self) -> str:
        return f"{super().cache_key}|task={self.task}"

    def describe(self) -> dict:
        d = super().describe()
        d.update(task=self.task, benchmark="PMO", score_range=[0.0, 1.0])
        return d

    def _evaluate_batch(self, smiles: Sequence[str]) -> Sequence[float | None]:
        try:
            raw = self._tdc(list(smiles))
        except Exception:
            # A whole-batch failure must not be recorded as a batch of zeros --
            # that would look like 'the method proposed bad molecules'.
            return [None] * len(smiles)
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        out: list[float | None] = []
        for v in raw:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(None)
        return out


PROBE_SMILES: tuple[str, ...] = (
    "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",  # haloperidol, a D2 ligand
    "CC(=O)Oc1ccccc1C(=O)O",                          # aspirin
    "c1ccc2ncnc(Nc3ccccc3)c2c1",                      # anilinoquinazoline
    "CCO",                                            # ethanol
    "CN1c2ccccc2C2(C1=O)c1c(ccc3ccccc13)-c1ccccc1N2",
)
"""Structurally unrelated molecules spanning drug-like and trivial. Any working
oracle separates at least two of them."""


def probe_oracle(task: str) -> tuple[str, str]:
    """Return ``(status, detail)`` for ``task`` before a budget is spent on it.

    ``status`` is one of:

    ``ok``
        The oracle separates unrelated molecules. Run it.
    ``broken``
        It raises, or returns ``None`` for everything. On a current
        scikit-learn, TDC's pickled ``drd2`` and ``gsk3b`` models load without
        complaint and then score **every** molecule identically --- raw TDC
        scores haloperidol, a known D2 ligand, at 0.0 --- while ``jnk3`` raises
        outright. The campaign otherwise spends its whole budget, every arm ties
        at exactly 0.0000, and the task silently drags every cross-task mean
        toward zero while compressing the differences between arms. Skip it.
    ``flat``
        It works, but returns one value across the probes. ``valsartan_smarts``
        does this legitimately: the SMARTS matches essentially nothing in
        ZINC-250k, which is a hard task rather than a broken one. **Run it** ---
        it is a real PMO task and dropping it would quietly change the
        benchmark --- but mark it, because a mean that includes a task where
        every method scores zero is not comparable to one that does not.

    "Every arm tied", "the task is impossible" and "the oracle was broken" are
    three different claims, and a results file must not render them the same.
    """
    try:
        oracle = PMOOracle(task, budget=None)
    except Exception as exc:  # pragma: no cover - environment dependent
        return "broken", f"{type(exc).__name__}: {str(exc).splitlines()[0][:150]}"

    try:
        scores = [oracle(smi) for smi in PROBE_SMILES]
    except Exception as exc:  # pragma: no cover - environment dependent
        return "broken", f"scoring raised {type(exc).__name__}: {str(exc).splitlines()[0][:130]}"

    flat = [v[0] if isinstance(v, list) else v for v in scores]
    usable = [v for v in flat if v is not None]
    if not usable:
        return "broken", "oracle returned None for every probe molecule"
    if len(usable) < len(flat):
        return "broken", f"oracle returned None for {len(flat) - len(usable)} of {len(flat)} probes"
    if len(set(usable)) == 1:
        return "flat", f"identical score ({usable[0]}) for {len(usable)} unrelated molecules"
    return "ok", f"{len(set(usable))} distinct scores across {len(flat)} probes"


def available_tasks(verbose: bool = False) -> list[str]:
    """Which PMO tasks actually load in this environment.

    The three ML-based tasks (drd2, gsk3b, jnk3) unpickle scikit-learn models
    and can fail on a newer sklearn than they were saved with; the rest are
    pure RDKit.
    """
    ok = []
    for task in PMO_TASKS:
        try:
            PMOOracle(task, budget=None)("CCO")
            ok.append(task)
        except Exception as exc:  # pragma: no cover - environment dependent
            if verbose:
                print(f"  {task:26s} unavailable: {str(exc)[:70]}")
    return ok
