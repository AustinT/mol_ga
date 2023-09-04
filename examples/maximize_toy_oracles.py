"""Maximize any number of toy oracle functions."""

from __future__ import annotations

import argparse
import logging
import time

import joblib
import numpy as np

try:
    from tdc import Oracle
except ImportError:
    print("This script requires the TDC package to be installed.")
    exit()

from mol_ga.mol_libraries import random_zinc
from mol_ga.preconfigured_gas import default_ga

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=str, default="qed")
    parser.add_argument("--offspring_size", type=int, default=1000)
    parser.add_argument("--max_generations", type=int, default=100)
    parser.add_argument("--start_population_size", type=int, default=10_000)
    args = parser.parse_args()
    logging.info(f"Arguments: {args}")
    logging.info(f"Running GA to maximize {args.oracle}...")

    oracle = Oracle(name=args.oracle)

    start_time = time.monotonic()
    with joblib.Parallel(n_jobs=-1) as parallel:
        output = default_ga(
            starting_population_smiles=random_zinc(args.start_population_size),
            scoring_function=oracle,
            max_generations=args.max_generations,
            offspring_size=args.offspring_size,
            parallel=parallel,
        )
    end_time = time.monotonic()
    top_100_scores = sorted([score for score, _ in output.population], reverse=True)[:100]
    print("Top few scores:")
    print(top_100_scores[:25])
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    guacamol_top1_10_100_score = (top_100_scores[0] + np.mean(top_100_scores[:10]) + np.mean(top_100_scores[:100])) / 3
    print(
        f"Average of top 1, top 10, and top 100 scores (reported metric in Guacamol): {guacamol_top1_10_100_score:.2f}"
    )
