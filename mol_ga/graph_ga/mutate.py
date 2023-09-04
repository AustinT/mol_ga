from __future__ import annotations

from random import Random
from typing import Optional

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from . import crossover as co

rdBase.DisableLog("rdApp.error")


def delete_atom(rng: Random):
    choices = [
        "[*:1]~[D1:2]>>[*:1]",
        "[*:1]~[D2:2]~[*:3]>>[*:1]-[*:3]",
        "[*:1]~[D3:2](~[*;!H0:3])~[*:4]>>[*:1]-[*:3]-[*:4]",
        "[*:1]~[D4:2](~[*;!H0:3])(~[*;!H0:4])~[*:5]>>[*:1]-[*:3]-[*:4]-[*:5]",
        "[*:1]~[D4:2](~[*;!H0;!H1:3])(~[*:4])~[*:5]>>[*:1]-[*:3](-[*:4])-[*:5]",
    ]
    p = [0.25, 0.25, 0.25, 0.1875, 0.0625]

    return rng.choices(choices, weights=p)[0]


def append_atom(rng: Random):
    choices = [
        ["single", ["C", "N", "O", "F", "S", "Cl", "Br"], 7 * [1.0 / 7.0]],
        ["double", ["C", "N", "O"], 3 * [1.0 / 3.0]],
        ["triple", ["C", "N"], 2 * [1.0 / 2.0]],
    ]
    p_BO = [0.60, 0.35, 0.05]

    index = rng.choices(list(range(3)), weights=p_BO)[0]

    BO, atom_list, p = choices[index]
    new_atom = rng.choices(atom_list, weights=p)[0]  # type: ignore[arg-type]  # type of p unclear

    if BO == "single":
        rxn_smarts = "[*;!H0:1]>>[*:1]X".replace("X", "-" + new_atom)  # type: ignore[operator]
    if BO == "double":
        rxn_smarts = "[*;!H0;!H1:1]>>[*:1]X".replace("X", "=" + new_atom)  # type: ignore[operator]
    if BO == "triple":
        rxn_smarts = "[*;H3:1]>>[*:1]X".replace("X", "#" + new_atom)  # type: ignore[operator]

    return rxn_smarts


def insert_atom(rng: Random):
    choices = [
        ["single", ["C", "N", "O", "S"], 4 * [1.0 / 4.0]],
        ["double", ["C", "N"], 2 * [1.0 / 2.0]],
        ["triple", ["C"], [1.0]],
    ]
    p_BO = [0.60, 0.35, 0.05]

    index = rng.choices(list(range(3)), weights=p_BO)[0]

    BO, atom_list, p = choices[index]
    new_atom = rng.choices(atom_list, weights=p)[0]  # type: ignore[arg-type]  # type of p unclear

    if BO == "single":
        rxn_smarts = "[*:1]~[*:2]>>[*:1]X[*:2]".replace("X", new_atom)  # type: ignore[arg-type]
    if BO == "double":
        rxn_smarts = "[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]".replace("X", new_atom)  # type: ignore[arg-type]
    if BO == "triple":
        rxn_smarts = "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]".replace("X", new_atom)  # type: ignore[arg-type]

    return rxn_smarts


def change_bond_order(rng: Random):
    choices = [
        "[*:1]!-[*:2]>>[*:1]-[*:2]",
        "[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]",
        "[*:1]#[*:2]>>[*:1]=[*:2]",
        "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]",
    ]
    p = [0.45, 0.45, 0.05, 0.05]

    return rng.choices(choices, weights=p)[0]


def delete_cyclic_bond():
    return "[*:1]@[*:2]>>([*:1].[*:2])"


def add_ring(rng: Random):
    choices = [
        "[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1",
        "[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1",
    ]
    p = [0.05, 0.05, 0.45, 0.45]

    return rng.choices(choices, weights=p)[0]


def change_atom(mol, rng: Random):
    choices = ["#6", "#7", "#8", "#9", "#16", "#17", "#35"]
    p = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]

    X = rng.choices(choices, weights=p)[0]
    _loop_count = 0
    while not mol.HasSubstructMatch(Chem.MolFromSmarts("[" + X + "]")):
        X = rng.choices(choices, weights=p)[0]

        # Avoid infinite loop potential here
        _loop_count += 1
        if _loop_count > 100:
            break

    Y = rng.choices(choices, weights=p)[0]
    while Y == X:
        Y = rng.choices(choices, weights=p)[0]

    return "[X:1]>>[Y:1]".replace("X", X).replace("Y", Y)


def mutate(mol: Chem.Mol, rng: Random) -> Optional[Chem.Mol]:
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        return mol

    p = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]
    for _ in range(10):
        rxn_smarts_list = 7 * [""]
        rxn_smarts_list[0] = insert_atom(rng)
        rxn_smarts_list[1] = change_bond_order(rng)
        rxn_smarts_list[2] = delete_cyclic_bond()
        rxn_smarts_list[3] = add_ring(rng)
        rxn_smarts_list[4] = delete_atom(rng)
        rxn_smarts_list[5] = change_atom(mol, rng)
        rxn_smarts_list[6] = append_atom(rng)
        rxn_smarts = rng.choices(rxn_smarts_list, weights=p)[0]

        rxn = AllChem.ReactionFromSmarts(rxn_smarts)

        new_mol_trial = rxn.RunReactants((mol,))

        new_mols = []
        for m in new_mol_trial:
            m = m[0]
            if co.mol_ok(m, rng) and co.ring_OK(m):
                new_mols.append(m)

        if len(new_mols) > 0:
            return rng.choice(new_mols)

    return None
