from __future__ import annotations

import random
from pathlib import Path
from typing import Optional


def random_zinc(size: int, rng: Optional[random.Random] = None) -> list[str]:
    zinc_path = Path(__file__).parent / "data" / "zinc250k.smiles"
    assert zinc_path.exists() and zinc_path.is_file()
    with open(zinc_path, "r") as f:
        all_zinc_smiles = f.readlines()
    
    # Remove whitespace and remove empty strings
    all_zinc_smiles = [s.strip() for s in all_zinc_smiles]
    all_zinc_smiles = [s for s in all_zinc_smiles if s]

    # Return random sample
    rng = rng or random.Random()
    return rng.choices(population=all_zinc_smiles, k=size)