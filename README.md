# Molecule Genetic Algorithms (mol_ga)

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Common Changelog](https://common-changelog.org/badge.svg)](https://common-changelog.org)

A simple, lightweight python package for genetic algorithms on molecules.
Key features:

- 📦 Works out-of-the-box (see [example](#quick-example) below).
- 🛠‍ Modular design: easy to extend/override default behaviour.
- 🧑‍🤝‍🧑 Batch-native: always calls objective function _in batches_ for flexible parallelization.
- ⚙️ Minimal dependencies. Should be compatible with nearly all python environments.
- ⛓️ Supports parallel mutation/crossover with `joblib`.
- 🎓 Smart caching of objective function (avoids duplicate calls).
- ⏯️ Stateless: can interleave GA steps with other control logic without side effects

## Installation

Install from [PyPI](https://pypi.org/project/mol-ga/):

```bash
pip install mol_ga
```

Alternatively, to install the latest version from GitHub, run

```bash
pip install https://github.com/AustinT/mol_ga.git
```

## Why mol_ga?

In my research, I've found GAs to be pretty powerful algorithms for molecular optimization,
but found problems in open-source implementations.
With `mol_ga` I hope to fill the niche for a lightweight, standalone GA implementation
that researchers can use quickly.

## Library design

There are countless variations of genetic algorithms.
This library provides a fairly general implementation in `general_ga.py`
and sensible defaults in `mol_ga.default_ga`.
The abstract GA consists of the following steps:

1. Sampling members from a "population" of SMILES (_default_: sample uniformly from top quantiles)
2. Use the samples from step 1 to produce new candidate SMILES (_default_: use Graph GA mutation/crossover operators)
3. Score the new candidate SMILES with the objective function and add them to the population.
4. Shrink the population if it is too large (_default_: choose the molecules with the highest score)

This design lets you run something reasonable in just a few lines of code,
while also allowing you to use more advanced features such as:

- Mutating SMILES with something else (e.g. convert to SELFIES string and add/delete/modify random token): override default from step 2.
- Fancy population selection (e.g. preserving diversity, accounting for constraints): override step 4 (and maybe step 1)

## Quick Example

Using this library is simple:
specify the objective function, the starting population
(or use our defaults),
and a few parameters.

```python
import joblib
from rdkit import Chem
from rdkit.Chem import QED

from mol_ga import mol_libraries, default_ga

# Function to optimize: we choose QED.
# mol_ga is designed for batch functions so it inputs a list of SMILES and outputs a list of floats.
f_opt = lambda s_list: [QED.qed(Chem.MolFromSmiles(s)) for s in s_list]

# Starting molecules: we choose random molecules from ZINC
# (we provide an easy handle for this)
start_smiles = mol_libraries.random_zinc(1000)

# Run GA with fast parallel generation
with joblib.Parallel(n_jobs=-1) as parallel:
    ga_results = default_ga(
        starting_population_smiles=start_smiles,
        scoring_function=f_opt,
        max_generations=100,
        offspring_size=100,
        parallel=parallel,
    )

# Print the best molecule
print(max(ga_results.population))
```

Output (remember it is random so results will vary between runs and between machines):

`(0.948131065160597, 'C[C@H]1CCN(C(=O)C(c2ccccc2)c2ccccc2)O[C@H]1N')`

## Citation

To make this library, I mainly drew on the algorithm from Jensen (2019)
and the code from the GuacaMol paper.
Please cite these papers if you use this library in an academic publication:

```
@article{jensen2019graph,
  title={A graph-based genetic algorithm and generative model/Monte Carlo tree search for the exploration of chemical space},
  author={Jensen, Jan H},
  journal={Chemical science},
  volume={10},
  number={12},
  pages={3567--3572},
  year={2019},
  publisher={Royal Society of Chemistry}
}
@article{brown2019guacamol,
  title={GuacaMol: benchmarking models for de novo molecular design},
  author={Brown, Nathan and Fiscato, Marco and Segler, Marwin HS and Vaucher, Alain C},
  journal={Journal of chemical information and modeling},
  volume={59},
  number={3},
  pages={1096--1108},
  year={2019},
  publisher={ACS Publications}
}
```

I first used this library in a
[benchmarking paper on arXiv](https://arxiv.org/abs/2310.09267).
You can also cite this paper.

## Contributing

This is an open-source project and PRs are welcome!

**!!PLEASE READ THIS SECTION BEFORE MAKING A PR!!**

More information is in `DEVELOPMENT.md`.

### Pre-commit

Use pre-commit to enforce formatting, large file checks, etc.
**Please set this up if you will add code to this repository**.

```bash
conda install pre-commit  # only if not installed already
pre-commit install
```

Now a series of useful checks will be run before any commit.

### Testing

We use `pytest` to test the codebase.
Everything can be run with the following line of code:

```bash
python -m pytest
```

Please make sure tests pass on your PR.

## Misc Information

### Are GraphGAs useful for my problem?

If your problem involves generating molecules in 2D, I think the answer is almost certainly *yes*.
I think that most papers in this space should run GraphGA as a baseline,
and that many researchers unfairly dismiss GAs because they don't understand them very well.
Here are my responses to various objections to GAs:

- _"GAs are not sample-efficient"_: in general I agree, but [Gao et al](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html) showed that they are surprisingly competitive with supposedly more sample-efficient methods. Nonetheless, GAs definitely **can** be part of a sample-efficient optimization pipeline: for example, they can be used to optimize the acquisition function for Bayesian optimization.
- _"I care about diversity, and GAs do not produce diverse outputs"_: most algorithms don't explicitly optimize for diversity, and therefore produce diversity "by accident". Graph GAs are no different in this regard. You don't know how diverse your outputs will be until you try it.
- _"I am doing mult-objective optimization"_: try scalarizing your problem with multiple sets of weights to explore the space of trade-offs.
- _"I am just trying to generate novel molecules without any objective function."_: just use a constant objective function and the GA will continually produce new molecules.

### Assumptions made implicitly by this package

Generally this package assumes that:

1. Objective function is **deterministic** (always outputs the same value for the same SMILES).
   If the function passed in is not deterministic, the GA will end up using the first value it returns.
2. Objective function is **scalar-valued** (I may add support for non-scalar objectives in the future).
3. Value is the **same for all SMILES corresponding to the same molecule**.
   Internally the algorithm will generally canonlicalize all SMILES,
   but it is possible for something to slip through.
   The GA might behave weirdly if the function returns different values for different SMILES
   strings for the same molecule.

If any of these assumptions are not satisfied, this package might still run without crashing,
but you could see unexpected behavior.

### What do you mean by "stateless"?

Under mild assumptions you can call the general `run_ga_maximization` function
with `max_generations=N` and the effect
will be the same as calling it `N` times
with `max_generations=1`. This allows you to add extra logic to the GA loop without
modifying the library (even though it might be a bit awkward).
See [this GitHub issue](https://github.com/AustinT/mol_ga/issues/11) for an
example.

Truly stateless behavior is not 100% guaranteed though. At the very least, you would need the following:

1. The starting population needs to be set to the ending population from the previous GA round.
2. The same random number generator should be passed in
3. The population selection function should be "idempotent":
  ie calling it twice is the same as calling it once.
  This is necessary because calling `run_ga_maximization` in a loop will cause it to be called twice.
  Making the population selection function idempotent is less difficult than it may seem though:
  any function which will not change the population if the size is below the maximum
  will be idempotent.
4. You must wrap the objective function in a `CachedBatchFunction`

### Why the emphasis on batched evaluation?

Many objective functions can be evaluated more easily in a batched setting, such as:

- Outputs of a machine learning model (neural network).
- Performing expensive simulations (they could be done in parallel)

By ensuring that the GA calls the objective function in batches whenever possible
it enables maximum speedup from batching.
However, the _downside_ of this is users are responsible for their own batching:
for example, if your ML model can only operate in batches of size <= 32,
then _you_ are responsible for breaking down a batch of possibly more than 32 molecules
into chunks of size <= 32.

### How does the default GA differ from the GraphGA baseline in GuacaMol?

I've tried to make many improvements:

- Fix [bug](https://github.com/BenevolentAI/guacamol_baselines/issues/11) where certain molecules can be "lost" from the population during mutation.
- Randomness can be controlled by passing a `random.Random` object (their implementation just uses the system random)
- Changed population selection to a method based on _quantiles_ rather than weighted sampling, using the objective function values as the weights. The original method had a few disadvantages which are resolved by sampling based on quantiles:
  1. Doesn't work for negative objective functions.
  2. If just one point with a high-score is found, it can easily dominate the sampling.
  3. If the population size is too high, the best molecules get "crowded out" by less good ones.
- Because of the sampling change above, it can handle large population sizes (e.g. 10k). This helps prevent the population collapsing to a bunch of similar molecules.
- Removed some [weird lines of code](https://github.com/BenevolentAI/guacamol_baselines/blob/44d24c53f3acf9266eb2fb06dbff909836549291/graph_ga/crossover.py#L70-L84) that effectively prevents the algorithm from producing molecules with more than 50 atoms.
  While it is true that most drugs are not this large, the algorithm should not be inherently incapable of producing molecules of more than a certain size!

### Limitations / known issues

In no particular order:

- **Synthesizability**: there is no guarantee that the molecules produced will be
  synthesizable (or even "nice"). If you don't like the molecules that you see
  you will need to either 1) change the scoring function or 2) write your own
  offspring generation method which avoids generating poor molecules.
- **Slow for large molecules**: the mutation operations in the default GA
  are more expensive for large molecules, which can cause the GA to run
  extremely slowly. If you encounter this, one option is to add a penalty
  for molecule size into the scoring function.

## Development status

At the moment [@austint] is the sole maintainer of this package.
Currently there are no plans to add new functionality, although
PRs are welcome, and [@austint] will try to address any bugs
or issues in a timely manner.

[@austint]: https://github.com/AustinT
