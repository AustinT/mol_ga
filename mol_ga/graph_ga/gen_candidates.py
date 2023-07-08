import logging

from rdkit import Chem, RDLogger

from . import crossover as co, mutate as mu


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
