""" Main code for running GAs. """
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pprint import pformat
from typing import Any, Callable, Optional, Union

import joblib
import numpy as np

from .cached_function import CachedBatchFunction

# Logger with standard handler
ga_logger = logging.getLogger(__name__)


@dataclass
class GAResults:
    """Results from a GA run."""

    population: list[tuple[float, str]]
    scoring_func_evals: dict[str, float]
    gen_info: list[dict[str, Any]]


def run_ga_maximization(
    *,
    scoring_func: Union[Callable[[list[str]], list[float]], CachedBatchFunction],
    starting_population_smiles: set[str],
    sampling_func: Callable[[list[tuple[float, str]], int, random.Random], list[str]],
    offspring_gen_func: Callable[[list[str], int, random.Random, Optional[joblib.Parallel]], set[str]],
    selection_func: Callable[[int, list[tuple[float, str]]], list[tuple[float, str]]],
    max_generations: int,
    population_size: int,
    offspring_size: int,
    rng: Optional[random.Random] = None,
    num_samples_per_generation: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    parallel: Optional[joblib.Parallel] = None,
) -> GAResults:
    """
    Runs a genetic algorithm to maximize `scoring_func`.

    Args:
        scoring_func: Function that takes a list of SMILES and returns a list of scores.
            Convention: scoring functionis being MAXIMIZED!
        starting_population_smiles: Set of SMILES to start the GA with.
        sampling_func: Function that takes a list of (score, smiles) tuples and returns a list of SMILES.
            This is used to sample SMILES from the population to create offspring.
        offspring_gen_func: Function that takes a list of SMILES and returns a list of SMILES.
            This can be anything (e.g. crossover, mutation).
        selection_func: Function to select the new population.
            This can be anything from "take the N best scores" to some more complicated method to pick
            a diverse subset.
        max_generations: Maximum number of generations to run the GA for.
        population_size: Number of SMILES to keep in the population.
        offspring_size: Number of offspring to create per generation.
        rng: Random number generator.
        num_samples_per_generation: Number of samples to take from the population to create offspring.
        logger: Logger to use.
        parallel: Joblib parallel object to use (to generate offspring in parallel).

    Returns:
        GAResults object containing the population, scoring function, and information about each generation.
    """

    # ============================================================
    # 0: Process input variables
    # ============================================================
    logger = logger or ga_logger
    logger.info("Starting GA maximization...")
    num_samples_per_generation = num_samples_per_generation or 2 * offspring_size
    rng = rng or random.Random()

    # Create the cached scoring function
    if not isinstance(scoring_func, CachedBatchFunction):
        scoring_func = CachedBatchFunction(scoring_func)
    start_cache_size = len(scoring_func.cache)
    logger.debug(f"Starting cache made, has size {start_cache_size}")

    # ============================================================
    # 1: prepare initial population
    # ============================================================

    # Score initial SMILES
    population_smiles = list(starting_population_smiles)
    population_scores = scoring_func.eval_batch(population_smiles)
    _starting_max_score = max(population_scores)
    logger.debug(f"Initial population scoring done. Pop size={len(population_smiles)}, Max={_starting_max_score}")
    population = list(zip(population_scores, population_smiles))
    del population_scores, population_smiles, _starting_max_score

    # Perform initial selection
    population = selection_func(population_size, population)

    # ============================================================
    # 2: run GA iterations
    # ============================================================

    # Run GA
    gen_info: list[dict[str, Any]] = []
    for generation in range(max_generations):
        logger.info(f"Start generation {generation}")

        # Separate out into SMILES and scores
        _, population_smiles = tuple(zip(*population))  # type: ignore[assignment]

        # Sample SMILES from population to create offspring
        samples_from_population = sampling_func(population, num_samples_per_generation, rng)

        # Create the offspring
        offspring = offspring_gen_func(
            samples_from_population,
            offspring_size,
            rng,
            parallel,
        )

        # Add to population, ensuring uniqueness
        population_smiles = list(set(population_smiles) | offspring)  # type: ignore[misc]  # thinks var deleted
        logger.debug(f"\t{len(offspring)} created")
        logger.debug(f"\tNew population size = {len(population_smiles)}")
        del offspring

        # Score new population
        logger.debug("\tCalling scoring function...")
        population_scores = scoring_func.eval_batch(population_smiles)
        logger.debug(f"\tScoring done, best score now {max(population_scores)}.")

        # Select new population
        population = list(zip(population_scores, population_smiles))
        population = selection_func(population_size, population)

        # Log results of this generation
        population_scores, population_smiles = tuple(zip(*population))  # type: ignore[assignment]
        gen_stats_dict = dict(
            max=np.max(population_scores),
            avg=np.mean(population_scores),
            median=np.median(population_scores),
            min=np.min(population_scores),
            std=np.std(population_scores),
            size=len(population_scores),
            num_func_eval=len(scoring_func.cache) - start_cache_size,
        )
        logger.info("End of generation. Stats:\n" + pformat(gen_stats_dict))
        gen_info.append(gen_stats_dict)
        del population_scores, population_smiles

    # ============================================================
    # 3: Create return object
    # ============================================================
    logger.info("End of GA. Returning results.")
    return GAResults(population=population, scoring_func_evals=scoring_func.cache, gen_info=gen_info)
