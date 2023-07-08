""" Main code for running GAs. """
from __future__ import annotations
from typing import List, Optional, Tuple, Union, Callable, Set
import logging
import heapq

import numpy as np

from .cached_function import CachedBatchFunction, CachedFunction


# Logger with standard handler
ga_logger = logging.getLogger("genetic-algorithm")
if len(ga_logger.handlers) == 0:
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    ga_logger.addHandler(ch)


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
    patience: int = None,
    max_func_calls: int = None,
    y_transform: callable = None,  # only used if scoring function is not a cached function already
    logger: logging.Logger = None,
    smiles_filter: Union[callable, CachedFunction] = None,
    early_stop_func: callable = None,
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
    if logger is None:
        logger = ga_logger
    logger.info("Starting GA maximization...")
    num_population_samples_per_generation = num_population_samples_per_generation or offspring_size

    # Create the cached function
    if not isinstance(scoring_function, CachedBatchFunction):
        scoring_function = CachedBatchFunction(scoring_function, transform=y_transform)
    start_cache = dict(scoring_function.cache)
    start_cache_size = len(start_cache)
    logger.debug(f"Starting cache made, has size {start_cache_size}")

    # Create max func calls
    if max_func_calls is None:

        # The most it could possibly be
        max_func_calls = max_generations * offspring_size + len(
            starting_population_smiles
        )
        max_func_calls *= 1000  # just to be safe
    max_func_calls += start_cache_size  # since it will just be measured by cache size

    # SMILES filter (cached in case it is expensive)
    if smiles_filter is None or isinstance(smiles_filter, CachedFunction):
        pass  # no need to do anything
    else:
        smiles_filter = CachedFunction(smiles_filter)  # cache it
    if isinstance(smiles_filter, CachedFunction):
        smiles_filter = smiles_filter.eval_non_batch  # so I can use "filter" function

    # ============================================================
    # 1: prepare initial population
    # ============================================================

    # Ensure unique
    population_smiles = set(starting_population_smiles)

    # Ensure passes filters
    population_smiles = list(filter(smiles_filter, population_smiles))

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
    _curr_max_score = _starting_max_score = max(population_scores)
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
    early_stop = False
    num_no_change_gen = 0
    gen_info = []
    for generation in range(max_generations):

        # Skip everything if early stopping
        if early_stop:
            continue

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
        population_smiles = set(population_smiles) | offspring
        logger.debug(f"\tNew population size = {len(population_smiles)}")

        # Remove ones which don't pass filters
        population_smiles = list(filter(smiles_filter, population_smiles))

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

        # early stopping if population doesn't change
        if population_smiles == old_population_smiles:

            num_no_change_gen += 1
            logger.info(f"\tPopulation unchanged for {num_no_change_gen} generations")
            if patience is not None and num_no_change_gen > patience:
                logger.info(f"\tThis exceeds patience of {patience}. Terminating GA.")
                early_stop = True
                break
        else:
            num_no_change_gen = 0

        # early stopping if budget is reached
        if max_func_calls is not None and len(scoring_function.cache) >= max_func_calls:
            logger.info(
                f"\tBudget of {max_func_calls - start_cache_size} has been reached. Terminating..."
            )
            early_stop = True

        # Early stopping from custom function
        if early_stop_func is not None and early_stop_func(population):
            early_stop = True

        # Clear variables from this iteration
        _curr_max_score = max(population_scores)
        del population_scores, population_smiles

    # Return values
    logger.info("End of GA. Returning results.")
    return (
        population,  # current population
        scoring_function.cache,  # holds all known function values
        (
            gen_info,
            early_stop,
        ),  # GA logs
    )
