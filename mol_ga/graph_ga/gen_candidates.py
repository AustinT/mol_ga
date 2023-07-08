import logging
import random
from typing import List

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


def generate_mols_v1(
    samples: List[str],
    n_candidates: int,
    frac_graph_ga_mutate: float = 0.10,
    frac_graph_ga_basic_smiles: float = 0.15,
):
    """Generate candiates in a blended way between Graph GA and Graph GA mutate only."""
    # TODO: clean up this function a lot

    # Turn off logging
    rd_logger = RDLogger.logger()
    rd_logger.setLevel(RDLogger.CRITICAL)

    # Step 2: set of graph GA mutations (mutation only)
    func_list = []
    arg_tuple_list = []
    n_graph_ga_mutate = int(n_candidates * frac_graph_ga_mutate)
    samples1 = samples[:n_graph_ga_mutate]
    func_list.append(mutate)
    arg_tuple_list.append([(s,) for s in samples1])

    # Step 4: set of graph GA reproductions
    n_basic = 100
    samples2 = random.choices(BASIC_SMILES, k=n_basic)  # choose some basic SMILES
    samples2 += samples[n_graph_ga_mutate:]

    mutation_rates = random.choices(
        [1e-3, 1e-2, 1e-1], cum_weights=[40, 90, 100], k=len(samples2)
    )
    random.shuffle(samples2)  # shuffle all together
    samples3 = list(samples2)  # make a copy
    random.shuffle(samples3)  # shuffle again
    func_list.append(reproduce)
    arg_tuple_list.append(list(zip(samples3, samples2, mutation_rates)))

    # Step 5: run everything, possibly in parallel
    offspring = []
    for func, args in zip(func_list, arg_tuple_list):
        offspring += [func(*t) for t in args]

    # Step 6: post-process and return offspring
    offspring = set(offspring)
    offspring.discard(None)
    return offspring