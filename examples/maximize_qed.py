from __future__ import annotations
import logging
from rdkit import Chem
from rdkit.Chem import QED

from mol_ga.mol_libraries import random_zinc
from mol_ga.preconfigured_gas import default_ga


def qed_value(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return QED.qed(mol)

def batch_qed_value(smiles_list: list[str]) -> list[float]:
    return [qed_value(smiles) for smiles in smiles_list]

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    output = default_ga(
        starting_population_smiles=random_zinc(1000),
        scoring_function=batch_qed_value,
        max_generations=10,
        offspring_size=100,
    )
    top_scores = sorted([score for score, _ in output.population], reverse=True)[:25]
    print("top_scores")
    print(top_scores)
