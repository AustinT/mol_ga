from __future__ import annotations
from random import Random
from typing import Optional

import joblib
from rdkit import Chem, RDLogger

from . import crossover as co, mutate as mu


rd_logger = RDLogger.logger()


def reproduce(
    smiles1: str, smiles2: str, mutation_rate: float, rng: Random, crossover_kwargs: dict = None
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
    new_child = co.crossover(parent_a, parent_b, rng, **crossover_kwargs)
    if new_child is not None and rng.random() < mutation_rate:
        new_child = mu.mutate(new_child, rng)
    if new_child is not None:
        new_child = Chem.MolToSmiles(new_child, canonical=True)
    return new_child


def mutate(smiles: str, rng: Random) -> Optional[str]:
    """Performs Graph GA mutations on a SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    new_mol = mu.mutate(mol, rng)
    if new_mol is not None:
        new_mol = Chem.MolToSmiles(new_mol, canonical=True, isomericSmiles=True)
    return new_mol


def graph_ga_blended_generation(
    samples: list[str],
    n_candidates: int,
    rng: Random,
    parallel: Optional[joblib.Parallel] = None,
    frac_graph_ga_mutate: float = 0.10,
) -> set[str]:
    """
    Generate candidates with a blend between Graph GA crossover (with some mutation)
    and Graph GA mutate only. Some minimal functional group SMILES are also included.
    """

    # Turn off logging
    rd_logger = RDLogger.logger()
    rd_logger.setLevel(RDLogger.CRITICAL)

    # Step 1: divide samples into "mutate" and "reproduce" sets
    n_graph_ga_mutate = int(n_candidates * frac_graph_ga_mutate)
    samples_mutate = samples[:n_graph_ga_mutate]
    samples_crossover = samples[n_graph_ga_mutate:]

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
            joblib.delayed(reproduce)(s1, s2, crossover_mut_rate, rng) for s1, s2 in zip(samples_crossover[:n_crossover], crossover_pairs)
        )
    else:
        offspring += [reproduce(s1, s2, crossover_mut_rate, rng) for s1, s2 in zip(samples_crossover[:n_crossover], crossover_pairs)]

    # Step 4: post-process and return offspring
    offspring = set(offspring)
    offspring.discard(None)  # this sometimes is returned
    return offspring
