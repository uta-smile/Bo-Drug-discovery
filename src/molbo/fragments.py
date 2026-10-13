"""A menu of real substructures mined from the library, for the model to choose from.

Why this exists: asking a 4B model to *write* SMARTS does not work. Measured on
Qwen3.5-4B against the EGFR series, every pattern it emitted was syntactically
invalid (unbalanced ring closures), the constraint collapsed to a molecular
weight range, and the arm kept 89% of the library -- indistinguishable from no
constraint at all.

That is a confound, not a result. The question under test is whether a language
model's *chemistry judgment* can narrow a search space usefully; whether it can
type a ring-closure index correctly is a different and much less interesting
question. So the model is handed an enumerated menu of substructures that
actually occur in the library and returns indices into it.

Three properties this buys:

* **Nothing invalid can enter.** Every menu entry is a pattern RDKit already
  parsed, mined from real molecules. The SMARTS failure mode disappears.
* **Every choice is meaningful.** Each entry comes with its library coverage,
  so the model cannot pick something that matches nothing (or everything)
  without seeing that it is doing so.
* **The comparison stays honest.** The menu is built from the *library*, which
  the optimiser can already see -- no unobserved values leak into it. The
  random-subset control and the data-driven control are unaffected, so the
  arms remain comparable.

Free-form SMARTS is still accepted alongside the menu; it is validated and
dropped if unparseable, and the drop rate is reported.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from dataclasses import dataclass
from typing import Sequence

from rdkit import Chem

from .chem import murcko_scaffold, to_mol

# Standard medicinal-chemistry groups, pre-validated. These are things a
# chemist would actually name in an SAR hypothesis, and having them as fixed
# menu items means the model can express "needs a basic amine" without typing
# a pattern.
FUNCTIONAL_GROUPS: dict[str, str] = {
    "aromatic nitrogen": "[n]",
    "basic aliphatic amine": "[NX3;H2,H1,H0;!$(N[!#6]);!$(N=*);!$(N#*)]",
    "secondary amide": "[NX3][CX3](=[OX1])",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "nitrile": "[NX1]#[CX2]",
    "halogen": "[F,Cl,Br,I]",
    "fluorine": "[F]",
    "chlorine or bromine": "[Cl,Br]",
    "trifluoromethyl": "[CX4]([F])([F])[F]",
    "methoxy": "[OX2][CH3]",
    "ether oxygen": "[OX2]([#6])[#6]",
    "hydroxyl": "[OX2H]",
    "michael acceptor (acrylamide)": "[CX3]=[CX3][CX3](=[OX1])[NX3]",
    "terminal alkyne": "[CX2]#[CH]",
    "piperazine": "C1CNCCN1",
    "piperidine": "C1CCNCC1",
    "morpholine": "C1COCCN1",
    "sulfone": "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    "aniline nitrogen": "[NX3;H1,H0][c]",
    "fused bicyclic aromatic": "[a]1[a][a]2[a][a][a][a][a]2[a]1",
}


@dataclass(frozen=True)
class Fragment:
    """One menu entry: a validated pattern plus what it covers."""

    index: int
    label: str
    smarts: str
    count: int
    """How many library members contain it."""

    source: str
    """``scaffold``, ``ring_system`` or ``functional_group``."""

    @property
    def fraction(self) -> float:
        return self._fraction

    def line(self, library_size: int) -> str:
        pct = 100.0 * self.count / max(1, library_size)
        return f"  [{self.index:2d}] {self.label:<34s} matches {self.count:5d} ({pct:4.1f}%)"


def ring_systems(smiles: str) -> list[str]:
    """Every fused ring system in a molecule, as canonical SMILES.

    A congeneric series' SAR usually turns on the core ring system plus what
    hangs off it, so these make better menu entries than whole scaffolds when
    the series is large and its scaffolds are nearly unique.
    """
    mol = to_mol(smiles)
    if mol is None:
        return []
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    if not rings:
        return []

    # Merge rings that share atoms -- that is what "fused" means.
    merged: list[set[int]] = []
    for ring in rings:
        hits = [m for m in merged if m & ring]
        for h in hits:
            merged.remove(h)
            ring = ring | h
        merged.append(ring)

    out = []
    for system in merged:
        try:
            frag = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(system), canonical=True)
            if frag:
                out.append(frag)
        except Exception:  # pragma: no cover - odd valence cases
            continue
    return out


def _valid_query(smarts: str) -> bool:
    return Chem.MolFromSmarts(smarts) is not None


def build_menu(
    library: Sequence[str],
    *,
    max_entries: int = 28,
    min_fraction: float = 0.02,
    max_fraction: float = 0.90,
    include_groups: bool = True,
    mine_sample: int | None = None,
    mine_seed: int = 0,
) -> list[Fragment]:
    """Mine candidate substructures from ``library``, most discriminating first.

    Entries matching almost everything or almost nothing are dropped: a pattern
    covering 95% of the library cannot narrow it, and one covering 0.1% would
    starve the search. ``min_fraction``/``max_fraction`` set that window.

    Nothing here looks at any molecule's *value*, only at structure, so the
    menu leaks no information the optimiser did not already have.
    """
    n = len(library)
    lo = max(1, int(min_fraction * n))
    hi = int(max_fraction * n)

    mols = [to_mol(x) for x in library]

    # Candidate *mining* (ring perception and Murcko scaffolds on every member)
    # is the expensive half and only has to find patterns worth considering, so
    # on a large pool it runs on a random subsample. Candidate *counting* still
    # runs on the full library, which is what keeps the coverage figure shown to
    # the model true rather than estimated -- the property a previous version of
    # this code got wrong.
    if mine_sample is not None and mine_sample < n:
        idx = np.random.default_rng(mine_seed).choice(n, size=mine_sample, replace=False)
        mine_from = [library[i] for i in idx]
    else:
        mine_from = list(library)

    # 1. Gather candidates with a cheap approximate count, only to shortlist.
    #    Generic scaffolds are deliberately absent: MakeScaffoldGeneric maps
    #    benzene to C1CCCCC1, which as a *query* matches aliphatic rings and so
    #    matches none of the aromatic molecules it came from. Its occurrence
    #    count and its match count are simply different numbers.
    approx: Counter[str] = Counter()
    labels: dict[str, str] = {}
    sources: dict[str, str] = {}

    scaffolds: Counter[str] = Counter()
    systems: Counter[str] = Counter()
    for smi in mine_from:
        scaf = murcko_scaffold(smi)
        if scaf:
            scaffolds[scaf] += 1
        for rs in set(ring_systems(smi)):
            systems[rs] += 1

    for source, counter in (("ring_system", systems), ("scaffold", scaffolds)):
        for patt, c in counter.items():
            if patt in approx or not _valid_query(patt):
                continue
            approx[patt] = c
            labels[patt] = patt
            sources[patt] = source

    if include_groups:
        for label, smarts in FUNCTIONAL_GROUPS.items():
            if smarts in approx or not _valid_query(smarts):  # pragma: no cover
                continue
            approx[smarts] = n  # unknown until counted; keep it in the shortlist
            labels[smarts] = label
            sources[smarts] = "functional_group"

    # 2. Shortlist generously, then count the survivors with the *same* query
    #    used at match time. This makes the coverage figure shown to the model
    #    true by construction rather than by coincidence. A pattern whose
    #    stated coverage is a lie would have the model confidently require
    #    something that matches nothing, starving the search.
    shortlist = sorted(approx, key=lambda q: -approx[q])[: max(max_entries * 4, 60)]

    counts: Counter[str] = Counter()
    seen_matches: dict[bytes, str] = {}
    for patt in shortlist:
        query = Chem.MolFromSmarts(patt)
        if query is None:  # pragma: no cover - validated above
            continue
        hits = np.fromiter(
            (m is not None and m.HasSubstructMatch(query) for m in mols),
            dtype=bool, count=len(mols),
        )
        c = int(hits.sum())
        if not (lo <= c <= hi):
            continue
        # Deduplicate on the *match set*, not the pattern string: C1CCNCC1,
        # C1CCCNC1 and C1CNCCC1 are three spellings of piperidine and would
        # otherwise occupy three menu slots that all mean the same thing,
        # crowding out genuinely different chemistry. Ties keep the entry with
        # the friendlier label (a curated group name over a raw SMILES).
        key = hits.tobytes()
        prior = seen_matches.get(key)
        if prior is not None:
            if sources.get(patt) == "functional_group" and sources.get(prior) != "functional_group":
                counts.pop(prior, None)
                seen_matches[key] = patt
                counts[patt] = c
            continue
        seen_matches[key] = patt
        counts[patt] = c

    # Rank by how close to an even split the pattern is: a fragment matching
    # half the library carries the most information about which half is better,
    # which is exactly what a constraint has to decide.
    def informativeness(patt: str) -> float:
        f = counts[patt] / n
        return abs(f - 0.35)

    ordered = sorted(counts, key=informativeness)[:max_entries]
    ordered.sort(key=lambda p: -counts[p])

    menu = []
    for i, patt in enumerate(ordered):
        frag = Fragment(
            index=i,
            label=labels[patt],
            smarts=patt,
            count=counts[patt],
            source=sources[patt],
        )
        object.__setattr__(frag, "_fraction", counts[patt] / n)
        menu.append(frag)
    return menu


def render_menu(menu: Sequence[Fragment], library_size: int) -> str:
    if not menu:
        return "(no substructure menu available)"
    lines = ["Available substructures, with how much of the library each matches:", ""]
    lines += [f.line(library_size) for f in menu]
    return "\n".join(lines)


def menu_digest(menu: Sequence[Fragment]) -> str:
    """Short stable identifier for a menu, for the response cache key."""
    import hashlib

    blob = "|".join(f"{f.index}:{f.smarts}:{f.count}" for f in menu)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
