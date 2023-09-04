from random import Random
from typing import Optional

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

rdBase.DisableLog("rdApp.error")


def cut(mol, rng: Random):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[*]-;!@[*]")):
        return None

    bis = rng.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[*]-;!@[*]")))  # single bond not in ring

    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]

    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

    try:
        return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
    except ValueError:
        return None


def cut_ring(mol, rng: Random):
    for _ in range(10):
        if rng.random() < 0.5:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")):
                return None
            bis = rng.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")))
            bis = (
                (bis[0], bis[1]),
                (bis[2], bis[3]),
            )
        else:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")):
                return None
            bis = rng.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")))
            bis = (
                (bis[0], bis[1]),
                (bis[1], bis[2]),
            )

        bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]

        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)])

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            if len(fragments) == 2:
                return fragments
        except ValueError:
            return None

    return None


def ring_OK(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))

    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list])
    macro_cycle = max_cycle_length > 6

    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts("[r3,r4]=[r3,r4]"))

    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


def mol_ok(
    mol,
    rng: Random,
    min_num_atoms=1,
    mean_num_atoms=40.0,
    std_num_atoms=1e3,  # by default, no real limit on maximum mol size
):
    try:
        Chem.SanitizeMol(mol)
        target_size = rng.gauss(mean_num_atoms, std_num_atoms)
        if mol.GetNumAtoms() > min_num_atoms and mol.GetNumAtoms() < target_size:
            return True
        else:
            return False
    except ValueError:
        return False


def crossover_ring(parent_A, parent_B, rng: Random, **mol_ok_kwargs):
    ring_smarts = Chem.MolFromSmarts("[R]")
    if not parent_A.HasSubstructMatch(ring_smarts) and not parent_B.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = [
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]",
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]",
    ]
    rxn_smarts2 = [
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]",
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]",
    ]

    for i in range(10):
        fragments_A = cut_ring(parent_A, rng)
        fragments_B = cut_ring(parent_B, rng)

        if fragments_A is None or fragments_B is None:
            return None

        new_mol_trial = []  # type: ignore  # wants list type
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            new_mol_trial = []
            for fa in fragments_A:
                for fb in fragments_B:
                    new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])

        new_mols = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for m in new_mol_trial:
                m = m[0]
                if mol_ok(m, rng, **mol_ok_kwargs):
                    new_mols += list(rxn2.RunReactants((m,)))

        new_mols2 = []
        for m in new_mols:
            m = m[0]
            if mol_ok(m, rng=rng, **mol_ok_kwargs) and ring_OK(m):
                new_mols2.append(m)

        if len(new_mols2) > 0:
            return rng.choice(new_mols2)

    return None


def crossover_non_ring(parent_A, parent_B, rng: Random, **mol_ok_kwargs):
    for i in range(10):
        fragments_A = cut(parent_A, rng)
        fragments_B = cut(parent_B, rng)
        if fragments_A is None or fragments_B is None:
            return None
        rxn = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
        new_mol_trial = []
        for fa in fragments_A:
            for fb in fragments_B:
                new_mol_trial.append(rxn.RunReactants((fa, fb))[0])

        new_mols = []
        for mol in new_mol_trial:
            mol = mol[0]
            if mol_ok(mol, rng=rng, **mol_ok_kwargs):
                new_mols.append(mol)

        if len(new_mols) > 0:
            return rng.choice(new_mols)

    return None


def crossover(parent_A, parent_B, rng: Random, **mol_ok_kwargs) -> Optional[Chem.Mol]:
    parent_smiles = [Chem.MolToSmiles(parent_A), Chem.MolToSmiles(parent_B)]
    try:
        Chem.Kekulize(parent_A, clearAromaticFlags=True)
        Chem.Kekulize(parent_B, clearAromaticFlags=True)

    except ValueError:
        pass

    for _ in range(10):
        if rng.random() <= 0.5:
            new_mol = crossover_non_ring(parent_A, parent_B, rng=rng, **mol_ok_kwargs)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol
        else:
            new_mol = crossover_ring(parent_A, parent_B, rng=rng, **mol_ok_kwargs)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol

    return None
