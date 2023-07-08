from rdkit import Chem
from rdkit.Chem import QED

from mol_ga.mol_libraries import random_zinc
from mol_ga.preconfigured_gas import default_ga


def qed_value(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return QED.qed(mol)

if __name__ == "__main__":
    output = default_ga(
        starting_population_smiles=random_zinc(1000),
        scoring_function=qed_value,
        max_generations=10,
        offspring_size=100,
    )
    top_scores = sorted(output[1].values(), reverse=True)[:25]
    print("top_scores")
    print(top_scores)
