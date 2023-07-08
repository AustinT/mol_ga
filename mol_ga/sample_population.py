from typing import List, Set, Tuple
import math
import random

def subsample_sorted_population_v1(
    population: List[Tuple[float, str]],
    n_sample: int,
    n_sample_chunks: int = 10,
    shuffle: bool = True,
) -> Set[str]:
    """
    Sample uniformly from the top N molecules, where N is chosen randomly (generally small).
    """

    population.sort(reverse=True)
    population = [s for _, s in population]  # only SMILES

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
