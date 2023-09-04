from __future__ import annotations

import joblib
import pytest
from rdkit import Chem
from rdkit.Chem import QED

from mol_ga import default_ga
from mol_ga.mol_libraries import random_zinc


def qed_value(smiles: str) -> float:
    """Calculate QED value of a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1.0
    return QED.qed(mol)


def batch_qed(smiles: list[str]) -> list[float]:
    """Calculate QED values of a list of SMILES strings."""
    return [qed_value(s) for s in smiles]


@pytest.mark.parametrize("parallel", [False, True])
def test_smoke(parallel: bool):
    """Test that the default GA runs with and without parallel."""
    kwargs = dict(
        starting_population_smiles=random_zinc(1000),
        scoring_function=batch_qed,
        max_generations=10,
        offspring_size=100,
    )

    if parallel:
        with joblib.Parallel(n_jobs=-1) as parallel:
            output = default_ga(**kwargs, parallel=parallel)
    else:
        output = default_ga(**kwargs, parallel=None)

    assert max(output.population)[0] > 0.935
