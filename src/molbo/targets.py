"""Protein sequences for targets, cut down to the part a ligand actually binds.

ESM2 caps at 1026 positions and many drug targets are longer -- EGFR is 1210
residues, USP7 is 1102. Feeding the model a truncation chosen by position is
arbitrary and can cut the binding site out entirely.

Worse, taking the *first* annotated domain is actively wrong. USP7's first
domain is MATH (68-195), a protein-protein interaction module; its catalytic
USP domain is 214-521 and holds both active-site residues (Cys223, His464).
Inhibitors bind the second. So the rule here is to take **the domain containing
the annotated active sites**, falling back to the largest domain, then to a
centre truncation -- and to record which rule fired, because a sequence chosen
by a fallback deserves less trust than one chosen by annotation.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

UNIPROT = "https://rest.uniprot.org/uniprotkb/{}.json"
CHEMBL_TARGET = "https://www.ebi.ac.uk/chembl/api/data/target/{}.json"


@dataclass
class TargetSequence:
    accession: str
    name: str
    sequence: str
    full_length: int
    start: int
    end: int
    rule: str
    """How the region was chosen: ``active-site domain``, ``largest domain``,
    ``full`` or ``centre truncation``. Anything but the first two is a guess."""

    def to_dict(self) -> dict:
        return {"accession": self.accession, "name": self.name, "sequence": self.sequence,
                "full_length": self.full_length, "start": self.start, "end": self.end,
                "rule": self.rule, "length": len(self.sequence)}


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as fh:
        return json.loads(fh.read().decode())


def uniprot_for_chembl(target_chembl_id: str) -> str | None:
    """ChEMBL target id -> UniProt accession. ChEMBL does not serve sequences."""
    j = _get(CHEMBL_TARGET.format(target_chembl_id))
    for comp in j.get("target_components", []):
        if comp.get("accession"):
            return comp["accession"]
    return None


def binding_domain(accession: str, max_len: int = 1022, pad: int = 25) -> TargetSequence:
    """Fetch a sequence and cut it to the region a small molecule would bind."""
    j = _get(UNIPROT.format(accession))
    seq = j["sequence"]["value"]
    name = (j.get("proteinDescription", {}).get("recommendedName", {})
            .get("fullName", {}).get("value", accession))

    def span(f):
        return f["location"]["start"]["value"], f["location"]["end"]["value"]

    feats = j.get("features", [])
    domains = [span(f) for f in feats if f["type"] == "Domain"]
    actives = [span(f)[0] for f in feats if f["type"] == "Active site"]

    if len(seq) <= max_len:
        return TargetSequence(accession, name, seq, len(seq), 1, len(seq), "full")

    # The domain holding the catalytic machinery, not merely the first one.
    holding = [(s, e) for (s, e) in domains if any(s <= a <= e for a in actives)]
    if holding:
        s, e = max(holding, key=lambda se: se[1] - se[0])
        rule = "active-site domain"
    elif domains:
        s, e = max(domains, key=lambda se: se[1] - se[0])
        rule = "largest domain"
    else:
        mid = len(seq) // 2
        s, e = max(1, mid - max_len // 2), min(len(seq), mid + max_len // 2)
        rule = "centre truncation"

    a, b = max(1, s - pad), min(len(seq), e + pad)
    if b - a + 1 > max_len:                      # still too long: centre on the domain
        mid = (a + b) // 2
        a, b = max(1, mid - max_len // 2), min(len(seq), mid + max_len // 2)
        rule += " (trimmed)"
    return TargetSequence(accession, name, seq[a - 1:b], len(seq), a, b, rule)


def save(target: TargetSequence, out_dir: str | Path = "data/targets") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{target.accession}.txt").write_text(target.sequence, encoding="utf-8")
    (out / f"{target.accession}.json").write_text(
        json.dumps(target.to_dict(), indent=2), encoding="utf-8")
    return out / f"{target.accession}.txt"
