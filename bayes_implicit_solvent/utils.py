from functools import lru_cache

import numpy as np
from openeye import oechem
from simtk import unit


def smiles_to_mol(smiles):
    """Create an openeye OEGraphMol object from this smiles string"""
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smiles)
    return mol


def get_gbsa_force(system):
    """Find and return the first force that contains 'GBSA' in its name"""
    for force in system.getForces():
        if "GBSA" in force.__class__.__name__:
            return force
    raise (Exception('No GBSA force found'))


def get_nb_force(system):
    """Find and return the first force that contains 'NonbondedForce' in its name"""
    forces = system.getForces()
    for f in forces:
        if 'NonbondedForce' in f.__class__.__name__:
            return f
    raise (Exception('No nonbonded force found'))


def get_charges(system):
    """Find and return the partial charges of all particles in the system"""
    nb_force = get_nb_force(system)
    return np.array(
        [nb_force.getParticleParameters(i)[0] / unit.elementary_charge for i in range(nb_force.getNumParticles())])


def apply_per_particle_params_to_GB_force(radii, scale_factors, gb_force):
    """Given arrays of radii and scale_factors,
    each of length = gb_force.getNumParameters(),
    overwrite the per-particle radius and scalingFactor parameters of gb_force.
    (Retain charges.)
    """

    for i in range(len(radii)):
        charge = gb_force.getParticleParameters(i)[0]
        gb_force.setParticleParameters(index=i,
                                       charge=charge,
                                       radius=radii[i],
                                       scalingFactor=scale_factors[i],
                                       )


def mdtraj_to_list_of_unitted_snapshots(traj):
    """Create list of (n_atoms, 3) snapshots with simtk.units attached"""
    return [snapshot * unit.nanometer for snapshot in traj.xyz]


def npy_sample_path_to_unitted_snapshots(path_to_npy_samples):
    """Given a path to a .npy file containing xyz coordinates in units of nanometers,
    create a list of (n_atoms, 3) snapshots with simtk.units attached"""
    xyz = np.load(path_to_npy_samples)
    traj = [snapshot * unit.nanometer for snapshot in xyz]
    return traj

def smarts_to_subsearch(smarts):
    """Creates an oechem.OESubsearch object from a SMARTS pattern"""
    qmol = oechem.OEQMol()
    oechem.OEParseSmarts(qmol, smarts)
    subsearch = oechem.OESubSearch(qmol)
    return subsearch


def remove_top_right_spines(ax):
    """Aesthetic tweak of matplotlib axes"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def get_substructure_matches(mol, subsearch):
    mol_ = oechem.OEMol(mol)
    n_atoms = len(list(mol_.GetAtoms()))
    matches = np.zeros(n_atoms, dtype=bool)

    for match in subsearch.Match(mol_, False):
        match_atoms = match.GetTargetAtoms()
        match_patterns = match.GetPatternAtoms()
        for a, p in zip(match_atoms, match_patterns):
            if p.GetIdx() == 0:
                matches[a.GetIdx()] = True
    return matches

# TODO: Inspect cache hit / miss rate (and maybe also memory consumption) see if this is an appropriate choice
@lru_cache(maxsize=2 ** 12)
def cached_substructure_matches(mol, subsearch_string):
    subsearch = smarts_to_subsearch(subsearch_string)
    return get_substructure_matches(mol, subsearch)


def convert_to_unitd_array(unitd_quantities):
    """Given an iterable of Quantities in compatible units, make a numpy
    array of unitless scalars, then multiply the array by the unit"""
    u = unitd_quantities[0].unit
    return np.array([q.value_in_unit(u) for q in unitd_quantities]) * u
