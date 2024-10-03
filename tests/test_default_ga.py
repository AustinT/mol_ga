from __future__ import annotations

import math

import joblib
import pytest
from rdkit import Chem
from rdkit.Chem import QED, Crippen

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


def qed_logp_gmean(smiles: str) -> float:
    """Geometric mean of qed and logp, negative if logp is negative."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1e9
    logp_val = Crippen.MolLogP(mol)
    qed_val = QED.qed(mol)
    gmean = math.sqrt(abs(logp_val) * qed_val)
    if logp_val < 0:
        return -gmean
    else:
        return gmean


def batch_qed_logp_gmean(smiles: list[str]) -> list[float]:
    return [qed_logp_gmean(s) for s in smiles]


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


def test_improvement():
    """
    Test that the GA sufficiently improves an objective which is easy to
    optimize (combination of QED and logP)

    It should be slightly harder to optimize this than to purely optimize logP,
    but still easy enough to see quick improvement.
    """
    start_population = random_zinc(1000)
    output = default_ga(
        starting_population_smiles=list(start_population),
        scoring_function=batch_qed_logp_gmean,
        max_generations=50,
        offspring_size=25,
    )

    # Test is improvement over starting population.
    # Improvement should be at least over a small threshold
    best_value = max(output.population)[0]
    best_starting_value = max(output.scoring_func_evals[s] for s in start_population)
    assert best_value - best_starting_value >= 0.05  # small but non-trivial improvement


def test_empty_population():
    """
    Test that passing an empty population raises an error.
    """
    with pytest.raises(ValueError):
        default_ga(
            starting_population_smiles=[],
            scoring_function=batch_qed,
            max_generations=10,
            offspring_size=100,
        )
