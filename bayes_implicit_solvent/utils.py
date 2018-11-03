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
    return np.array([nb_force.getParticleParameters(i)[0] / unit.elementary_charge for i in range(nb_force.getNumParticles())])

def apply_radii_to_GB_force(radii, gb_force):
    """Given an array of radii, of length = gb_force.getNumParameters(),
    overwrite the per-particle radii parameters of gb_force.
    Retain charges, set scalingFactors to 1.0.
    """
    # TODO: Don't reset scalingFactor to 1.0.

    for i in range(len(radii)):
        charge = gb_force.getParticleParameters(i)[0]
        gb_force.setParticleParameters(index=i,
                                       charge=charge,
                                       radius=radii[i],
                                       scalingFactor=1.0,
                                       )


def mdtraj_to_list_of_unitted_snapshots(traj):
    """Create list of (n_atoms, 3) snapshots with simtk.units attached"""
    return [snapshot * unit.nanometer for snapshot in traj.xyz]


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


subsearch_cache = dict()
number_of_cache_misses = 0
number_of_cache_accesses = 0


def cached_substructure_matches(mol, subsearch_string):
    global number_of_cache_accesses
    global number_of_cache_misses
    global subsearch_cache

    number_of_cache_accesses += 1

    arg_tuple = (mol, subsearch_string)

    if arg_tuple not in subsearch_cache:
        subsearch = smarts_to_subsearch(subsearch_string)
        subsearch_cache[arg_tuple] = get_substructure_matches(mol, subsearch)
        number_of_cache_misses += 1

    return subsearch_cache[arg_tuple]
