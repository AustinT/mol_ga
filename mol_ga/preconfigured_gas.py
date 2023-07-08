from __future__ import annotations

from mol_ga.graph_ga.gen_candidates import generate_mols_v1
from mol_ga.sample_population import subsample_sorted_population_v1
from mol_ga.general_ga import run_ga_maximization

def default_ga(
    starting_population_smiles: list[str],
    scoring_function,
    max_generations: int,
    offspring_size: int,
    offspring_gen_func=generate_mols_v1,
    population_sampling_function=subsample_sorted_population_v1,
    population_size=10_000,
):
    return run_ga_maximization(
        starting_population_smiles=starting_population_smiles,
        scoring_function=scoring_function,
        offspring_gen_func=offspring_gen_func,
        population_sampling_function=population_sampling_function,
        max_generations=max_generations,
        population_size=population_size,
        offspring_size=offspring_size,
    )