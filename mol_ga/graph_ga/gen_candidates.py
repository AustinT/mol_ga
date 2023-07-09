from __future__ import annotations
import random

from rdkit import Chem, RDLogger

from . import crossover as co, mutate as mu
from ..mol_libraries import BASIC_SMILES


rd_logger = RDLogger.logger()


def reproduce(
    smiles1: str, smiles2: str, mutation_rate: float, crossover_kwargs: dict = None
):
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
    if crossover_kwargs is None:
        crossover_kwargs = dict()
    new_child = co.crossover(parent_a, parent_b, **crossover_kwargs)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    if new_child is not None:
        new_child = Chem.MolToSmiles(new_child, canonical=True)
    return new_child


def mutate(smiles: str):
    """Performs Graph GA mutations on a SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    new_mol = mu.mutate(mol, 1.0)  # mutate with certainty
    if new_mol is not None:
        new_mol = Chem.MolToSmiles(new_mol, canonical=True, isomericSmiles=True)
    return new_mol


def graph_ga_blended_generation(
    samples: list[str],
    n_candidates: int,
    frac_graph_ga_mutate: float = 0.10,
):
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
    offspring = [mutate(s) for s in samples_mutate]

    # Step 3: run crossover
    random.shuffle(samples_crossover)
    # run crossover with shuffled version of list + some basic SMILES strings with minimal functional groups
    crossover_pairs = list(samples_crossover) + random.choices(BASIC_SMILES, k=len(samples_crossover) // 3)  
    random.shuffle(crossover_pairs)
    mutation_rates = random.choices(
        [1e-3, 1e-2, 1e-1], cum_weights=[40, 90, 100], k=n_candidates - len(samples_mutate)  # this is the limiting list
    )
    offspring += [reproduce(s1, s2, mr) for s1, s2, mr in zip(samples_crossover, crossover_pairs, mutation_rates)]

    # Step 4: post-process and return offspring
    offspring = set(offspring)
    offspring.discard(None)  # this sometimes is returned
    return offspring
