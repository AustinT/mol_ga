""" Main code for running GAs. """
from __future__ import annotations
from typing import List, Optional, Tuple, Union, Callable, Set
import logging
import heapq

import numpy as np

from .cached_function import CachedBatchFunction


# Logger with standard handler
ga_logger = logging.getLogger(__name__)


def run_ga_maximization(
    *,
    starting_population_smiles: List[str],
    scoring_function: Union[Callable[[list[str]], list[float]], CachedBatchFunction],
    population_sampling_function: Callable[[List[Tuple[float, str]], int], List[str]],
    offspring_gen_func: Callable[[List[str], int], Set[str]],
    max_generations: int,
    population_size: int,
    offspring_size: int,
    num_population_samples_per_generation: Optional[int] = None,
):
    """
    Runs a genetic algorithm to MAXIMIZE a score function.

    It does accurate budgeting by tracking which function calls have been made already.
    Note that the function will always be called with canonical smiles.

    Notes:
    1. max func calls might be overshot by a little bit
    2. Does not track the order in which the SMILES were queried
    """

    # ============================================================
    # 0: Process input variables
    # ============================================================
    logger = ga_logger
    logger.info("Starting GA maximization...")
    num_population_samples_per_generation = num_population_samples_per_generation or offspring_size

    # Create the cached function
    if not isinstance(scoring_function, CachedBatchFunction):
        scoring_function = CachedBatchFunction(scoring_function, )
    start_cache = dict(scoring_function.cache)
    start_cache_size = len(start_cache)
    logger.debug(f"Starting cache made, has size {start_cache_size}")


    # ============================================================
    # 1: prepare initial population
    # ============================================================

    # Ensure unique
    population_smiles = list(set(starting_population_smiles))

    # Eval scores
    num_start_eval = len(set(population_smiles) - set(start_cache.keys()))
    logger.debug(
        "Scoring initial population. "
        f"{num_start_eval}/{len(population_smiles)} "
        f"({num_start_eval/len(population_smiles)*100:.1f}%) "
        "not in start cache and will need evaluation."
    )
    del num_start_eval  # not needed later
    population_scores = scoring_function.eval_batch(population_smiles)
    _starting_max_score = max(population_scores)
    logger.debug(
        f"Initial population scoring done. Pop size={len(population_smiles)}, Max={_starting_max_score}"
    )

    # Sort population largest to smallest
    population = list(zip(population_scores, population_smiles))
    population = heapq.nlargest(population_size, population)
    del population_scores, population_smiles

    # ============================================================
    # 2: run GA iterations
    # ============================================================

    # Run GA
    gen_info = []
    for generation in range(max_generations):

        logger.info(f"Start generation {generation}")

        # Separate out into SMILES and scores
        _, population_smiles = tuple(zip(*population))
        old_population_smiles = list(population_smiles)

        # Create offspring
        samples_from_population = population_sampling_function(
            population, num_population_samples_per_generation
        )
        offspring = offspring_gen_func(
            samples_from_population,
            offspring_size,
        )
        logger.debug(f"\t{len(offspring)} created")

        # Add to population
        population_smiles = list(set(population_smiles) | offspring)
        logger.debug(f"\tNew population size = {len(population_smiles)}")

        # Find out scores
        logger.debug("\tStarting function calls...")
        population_scores = scoring_function.eval_batch(population_smiles)
        logger.debug(f"\tScoring done, best score now {max(population_scores)}.")

        # Sort population largest to smallest
        population = list(zip(population_scores, population_smiles))
        population = heapq.nlargest(population_size, population)
        population_scores, population_smiles = tuple(zip(*population))

        # Record results of generation
        gen_stats_dict = dict(
            max=np.max(population_scores),
            avg=np.mean(population_scores),
            median=np.median(population_scores),
            min=np.min(population_scores),
            std=np.std(population_scores),
        )
        gen_stats_dict = {k: f"{v:.5e}" for k, v in gen_stats_dict.items()}
        gen_stats_dict.update(
            dict(
                size=len(population_scores),
                num_func_eval=len(scoring_function.cache) - start_cache_size,
            )
        )
        stats_str = " ".join(
            ["\tGen stats:\n"] + [f"{k}={v}" for k, v in gen_stats_dict.items()]
        )
        logger.info(stats_str)
        gen_info.append(dict(smiles=population_smiles, **gen_stats_dict))

        # Clear variables from this iteration
        del population_scores, population_smiles

    # Return values
    logger.info("End of GA. Returning results.")
    return (
        population,  # current population
        scoring_function.cache,  # holds all known function values
        (
            gen_info,
        ),  # GA logs
    )
