from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from rdkit import Chem

# Basic SMILES that contain different functional groups
BASIC_SMILES = [
    "CC=C",  # alkene,
    "CC#C",  # alkyne,
    "CC[OH]",  # alcohol
    "COC",  # ether
    "CC(=O)[OH]",  # carboxylic acid
    "CC=O",  # aldehyde
    "CC(=O)C",  # ketone
    "CC(=O)N",  # amide
    "CCN",  # amine
    "CC#N",  # nitrile
]


def random_zinc(size: int, rng: Optional[random.Random] = None) -> list[str]:
    """Return random SMILES from the ZINC250k dataset."""
    zinc_path = Path(__file__).parent / "data" / "zinc250k.smiles"
    assert zinc_path.exists() and zinc_path.is_file()
    with open(zinc_path, "r") as f:
        all_zinc_smiles = f.readlines()

    # Remove whitespace and remove empty strings
    all_zinc_smiles = [s.strip() for s in all_zinc_smiles]
    all_zinc_smiles = [s for s in all_zinc_smiles if s]

    # Sample random SMILES
    rng = rng or random.Random()
    chosen_smiles = rng.choices(population=all_zinc_smiles, k=size)

    # Last check: ensure that all SMILES are canonical
    canon_smiles = [Chem.CanonSmiles(s) for s in chosen_smiles]
    return [s for s in canon_smiles if s]  # exclude None
