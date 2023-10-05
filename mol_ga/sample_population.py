from __future__ import annotations

import math
from random import Random

import numpy as np


def uniform_qualitle_sampling(
    population: list[tuple[float, str]],
    n_sample: int,
    rng: Random,
    shuffle: bool = True,
) -> list[str]:
    """Sample SMILES by sampling uniformly from logarithmically spaced top-N."""

    samples: list[str] = []
    quantiles = 1 - np.logspace(-3, 0, 25)
    n_samples_per_quanitile = int(math.ceil(n_sample / len(quantiles)))
    for q in quantiles:
        score_threshold = np.quantile([s for s, _ in population], q)
        eligible_population = [smiles for score, smiles in population if score >= score_threshold]
        samples.extend(rng.choices(population=eligible_population, k=n_samples_per_quanitile))

    # Shuffle samples to decrease correlations between adjacent samples
    if shuffle:
        rng.shuffle(samples)

    return samples[:n_sample]  # in case there are slightly too many samples
