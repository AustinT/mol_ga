from typing import List, Set
import math
import random
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDLogger

from mol_ga.ga_main import run_ga_maximization
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


def subsample_sorted_population_v1(
    population: List[str],
    n_sample: int,
    n_sample_chunks: int = 10,
    shuffle: bool = True,
) -> Set[str]:
    """
    Sample uniformly from the top N molecules, where N is chosen randomly (generally small).

    Note: assumes that population is in sorted order!!!!!!!
    """

    top_n_list = [
        5,
        10,
        20,
        30,
        40,
        50,
        60,
        75,
        100,
        250,
        1000,
        10_000,
        len(population),
    ]

    # Do sampling
    chunk_size = int(math.ceil(n_sample / n_sample_chunks))
    samples = []
    for _ in range(n_sample_chunks):
        n = random.choice(top_n_list)
        samples.extend(random.choices(population=population[:n], k=chunk_size))

    # Shuffle samples to decrease correlations between adjacent samples
    if shuffle:
        random.shuffle(samples)

    return samples[:n_sample]  # in case there are slightly too many samples


def generate_mols_v1(
    population: List[str],
    n_candidates: int,
    frac_selfies: float = 0.15,
    frac_graph_ga_mutate: float = 0.10,
    frac_graph_ga_basic_smiles: float = 0.15,
):
    """Generate candiates in a blended way between Graph GA and Graph GA mutate only."""

    # Turn off logging
    rd_logger = RDLogger.logger()
    rd_logger.setLevel(RDLogger.CRITICAL)

    # Step 0: discard any Nones from the population
    population = list(filter(None, population))

    # Step 1: choose population to mutate for candidate generation
    n_sample_chunks = n_candidates
    if n_candidates >= 1000:
        n_sample_chunks = 50
    elif n_candidates >= 100:
        n_sample_chunks = 10
    samples1 = subsample_sorted_population_v1(
        population=population, n_sample=n_candidates, n_sample_chunks=n_sample_chunks
    )

    # Step 2: set of graph GA mutations (mutation only)
    func_list = []
    arg_tuple_list = []
    n_graph_ga_mutate = int(n_candidates * frac_graph_ga_mutate)
    curr_samples = samples1[:n_graph_ga_mutate]
    samples1 = samples1[n_graph_ga_mutate:]
    func_list.append(graph_ga.mutate)
    arg_tuple_list.append([(s,) for s in curr_samples])
    del n_graph_ga_mutate, curr_samples

    # Step 4: set of graph GA reproductions
    curr_samples = samples1  # all remaining samples
    n_basic = int(len(curr_samples) * frac_graph_ga_basic_smiles)

    samples2 = random.choices(BASIC_SMILES, k=n_basic)  # choose some basic SMILES
    samples2 += subsample_sorted_population_v1(
        population=population,
        n_sample=len(curr_samples) - n_basic,
        n_sample_chunks=n_sample_chunks,
        shuffle=False,
    )

    mutation_rates = random.choices(
        [1e-3, 1e-2, 1e-1], cum_weights=[40, 90, 100], k=len(curr_samples)
    )
    random.shuffle(samples2)  # shuffle all together
    func_list.append(graph_ga.reproduce)
    arg_tuple_list.append(list(zip(curr_samples, samples2, mutation_rates)))
    del curr_samples, n_basic, samples2, mutation_rates

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
        max_generations=10,
        population_size=1000,
        offspring_size=100,
    )
    top_scores = sorted(output[1].values(), reverse=True)[:25]
    print("top_scores")
    print(top_scores)
