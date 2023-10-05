from __future__ import annotations

from random import Random
from typing import Optional

import joblib
from rdkit import Chem, RDLogger

from . import crossover as co
from . import mutate as mu

rd_logger = RDLogger.logger()


def sanitize_mol_to_smiles(mol: Chem.Mol) -> Optional[str]:
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:  # unsure what exceptions are thrown
        return None


def reproduce(
    smiles1: str, smiles2: str, mutation_rate: float, rng: Random, crossover_kwargs: Optional[dict] = None
) -> Optional[str]:
    """

    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation

    Returns:

    """

    # turn off rdkit logging
    rd_logger.setLevel(RDLogger.CRITICAL)

    parent_a = Chem.MolFromSmiles(smiles1)
    parent_b = Chem.MolFromSmiles(smiles2)
    crossover_kwargs = crossover_kwargs or dict()
    try:
        new_child = co.crossover(parent_a, parent_b, rng, **crossover_kwargs)
    except RuntimeError:
        return None  # occasionally this happens due to internal rdkit errors
    if new_child is not None and rng.random() < mutation_rate:
        new_child = mu.mutate(new_child, rng)
    if new_child is None:
        return None
    else:
        return sanitize_mol_to_smiles(new_child)


def mutate(smiles: str, rng: Random) -> Optional[str]:
    """Performs Graph GA mutations on a SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    new_mol = mu.mutate(mol, rng)
    if new_mol is None:
        return None
    else:
        return sanitize_mol_to_smiles(new_mol)


def graph_ga_blended_generation(
    samples: list[str],
    n_candidates: int,
    rng: Random,
    parallel: Optional[joblib.Parallel] = None,
    frac_graph_ga_mutate: float = 0.10,
) -> set[str]:
    """
    Generate candidates with a blend between Graph GA crossover (with some mutation)
    and Graph GA mutate only.
    """

    # Turn off logging
    rd_logger = RDLogger.logger()
    rd_logger.setLevel(RDLogger.CRITICAL)

    # Step 1: divide samples into "mutate" and "reproduce" sets
    samples_mutate: list[str] = []
    samples_crossover: list[str] = []
    for s in samples:
        if rng.random() < frac_graph_ga_mutate:
            samples_mutate.append(s)
        else:
            samples_crossover.append(s)
    # Ensure there are not too many samples in the mutate set
    samples_mutate = samples_mutate[: int(n_candidates * frac_graph_ga_mutate + 1)]  # add one to avoid rounding errors

    # Step 2: run mutations
    if parallel:
        offspring = parallel(joblib.delayed(mutate)(s, rng) for s in samples_mutate)
    else:
        offspring = [mutate(s, rng) for s in samples_mutate]

    # Step 3: run crossover betweeen the crossover samples and a shuffled version of itself
    n_crossover = n_candidates - len(offspring)
    crossover_pairs = list(samples_crossover)
    rng.shuffle(crossover_pairs)
    crossover_mut_rate = 1e-2
    if parallel:
        offspring += parallel(
            joblib.delayed(reproduce)(s1, s2, crossover_mut_rate, rng)
            for s1, s2 in zip(samples_crossover[:n_crossover], crossover_pairs)
        )
    else:
        offspring += [
            reproduce(s1, s2, crossover_mut_rate, rng)
            for s1, s2 in zip(samples_crossover[:n_crossover], crossover_pairs)
        ]

    # Step 4: post-process and return offspring
    offspring = set(offspring)
    offspring.discard(None)  # this sometimes is returned
    return offspring
