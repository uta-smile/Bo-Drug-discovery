"""Deterministic molecule standardisation.

Every oracle memoises on the output of :func:`canonical_smiles`, and every
enumerated library is keyed by it.  The two *must* agree exactly, so the whole
pipeline funnels through this one function and pins its behaviour with
:data:`STANDARDIZER_VERSION`.  Bump that string if the transformation changes;
it is written into cache files so stale caches are detected rather than
silently mixed with fresh values.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

# RDKit is chatty on malformed input; generative arms produce a lot of that.
RDLogger.DisableLog("rdApp.*")

STANDARDIZER_VERSION = "v1"
"""Identifier for the standardisation recipe below, embedded in cache keys."""


@dataclass(frozen=True)
class StandardizeOptions:
    """Knobs for :func:`canonical_smiles`.

    The defaults are what the ChEMBL lookup oracle uses: drop counter-ions so a
    hydrochloride salt matches its free base, keep the charge state and
    stereochemistry as drawn because congeneric-series SAR routinely turns on
    both.
    """

    strip_salts: bool = True
    neutralize: bool = False
    keep_stereo: bool = True

    def key(self) -> str:
        return (
            f"{STANDARDIZER_VERSION}"
            f":salt={int(self.strip_salts)}"
            f":neut={int(self.neutralize)}"
            f":stereo={int(self.keep_stereo)}"
        )


DEFAULT_OPTS = StandardizeOptions()

_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def to_mol(smiles: str, opts: StandardizeOptions = DEFAULT_OPTS) -> Chem.Mol | None:
    """Parse and standardise, or return ``None`` if RDKit cannot read it."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        if opts.strip_salts:
            mol = _LARGEST_FRAGMENT.choose(mol)
        if opts.neutralize:
            mol = _UNCHARGER.uncharge(mol)
        if not opts.keep_stereo:
            Chem.RemoveStereochemistry(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol if mol is not None and mol.GetNumAtoms() > 0 else None


def canonical_smiles(
    smiles: str, opts: StandardizeOptions = DEFAULT_OPTS
) -> str | None:
    """Canonical, standardised SMILES, or ``None`` for unparseable input.

    This is the memoisation key for every oracle in the project.
    """
    mol = to_mol(smiles, opts)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=opts.keep_stereo)


@lru_cache(maxsize=200_000)
def canonical_smiles_cached(smiles: str, opts_key: str = DEFAULT_OPTS.key()) -> str | None:
    """Process-local memo over :func:`canonical_smiles` for the default options.

    Canonicalisation is pure, and BO over a discrete library re-canonicalises
    the same few thousand strings thousands of times.
    """
    if opts_key != DEFAULT_OPTS.key():  # pragma: no cover - guard against misuse
        raise ValueError("canonical_smiles_cached only memoises the default options")
    return canonical_smiles(smiles, DEFAULT_OPTS)


def is_valid(smiles: str) -> bool:
    return canonical_smiles_cached(smiles) is not None


def murcko_scaffold(smiles: str, generic: bool = False) -> str | None:
    """Bemis-Murcko scaffold, used to carve a congeneric series out of ChEMBL."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = to_mol(smiles)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


FINGERPRINT_VERSION = "v2-chiral"
"""Identifier for the fingerprint recipe below. Any cache keyed on fingerprints
must include it, or a representation change is silently mixed with stale data.

v1: ECFP4, 2048 bits, ``includeChirality=False`` (RDKit's default).
v2: chirality on. See :data:`INCLUDE_CHIRALITY` for why.
"""

INCLUDE_CHIRALITY = True
"""Distinguish stereoisomers in the fingerprint.

RDKit's Morgan generator ignores stereochemistry by default, which collapses
E/Z and R/S isomers onto identical bit vectors. Measured on the EGFR
anilinoquinazoline library (2,086 compounds), that made **133 pairs -- 204
molecules, ~10% of the library -- indistinguishable to the surrogate** while
their measured activities differed by up to 3.0 log units, a thousand-fold in
potency. A GP cannot do anything sensible with that: it is forced to average
the pair, which puts an irreducible noise floor under every prediction and
shows up as a large fitted ``eta`` early in a run.

Turning chirality on halves it (133 -> 65 pairs, 204 -> 94 molecules, worst gap
3.0 -> 2.0). The remainder are regioisomers that ECFP4 at radius 2 genuinely
cannot separate; fixing those needs a larger radius or a different descriptor,
not a flag.

The cost is a slightly sparser feature space -- stereocentres now consume
distinct bits -- which is the right trade when the series has real
stereochemical SAR, as this one does.
"""

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, includeChirality=INCLUDE_CHIRALITY
)


def morgan_fingerprint(smiles: str, as_numpy: bool = True):
    """ECFP4 (2048 bit), stereochemistry-aware. The surrogate GP's representation."""
    mol = to_mol(smiles)
    if mol is None:
        return None
    fp = _MORGAN_GEN.GetFingerprint(mol)
    if not as_numpy:
        return fp
    import numpy as np

    arr = np.zeros((2048,), dtype=np.uint8)
    from rdkit import DataStructs

    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
