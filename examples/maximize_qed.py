from typing import List, Set
import math
import random
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDLogger

from mol_ga.general_ga import run_ga_maximization
from mol_ga.sample_population import subsample_sorted_population_v1
from mol_ga import graph_ga

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




def generate_mols_v1(
    samples: List[str],
    n_candidates: int,
    frac_graph_ga_mutate: float = 0.10,
    frac_graph_ga_basic_smiles: float = 0.15,
):
    """Generate candiates in a blended way between Graph GA and Graph GA mutate only."""

    # Turn off logging
    rd_logger = RDLogger.logger()
    rd_logger.setLevel(RDLogger.CRITICAL)

    # Step 2: set of graph GA mutations (mutation only)
    func_list = []
    arg_tuple_list = []
    n_graph_ga_mutate = int(n_candidates * frac_graph_ga_mutate)
    samples1 = samples[:n_graph_ga_mutate]
    func_list.append(graph_ga.mutate)
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
    func_list.append(graph_ga.reproduce)
    arg_tuple_list.append(list(zip(samples3, samples2, mutation_rates)))

    # Step 5: run everything, possibly in parallel
    offspring = []
    for func, args in zip(func_list, arg_tuple_list):
        offspring += [func(*t) for t in args]

    # Step 6: post-process and return offspring
    offspring = set(offspring)
    offspring.discard(None)
    return offspring


def qed_value(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return QED.qed(mol)

if __name__ == "__main__":
    smiles = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    print(f"QED value of {smiles} is {qed_value(smiles)}")
    output = run_ga_maximization(
        starting_population_smiles=[smiles],
        scoring_function=qed_value,
        offspring_gen_func=generate_mols_v1,
        population_sampling_function=subsample_sorted_population_v1,
        max_generations=10,
        population_size=1000,
        offspring_size=100,
    )
    top_scores = sorted(output[1].values(), reverse=True)[:25]
    print("top_scores")
    print(top_scores)
