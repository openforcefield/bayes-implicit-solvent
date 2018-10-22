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
    """Find and return the first force that contains 'NonbondedForce in its name"""
    forces = system.getForces()
    for f in forces:
        if 'NonbondedForce' in f.__class__.__name__:
            return f
    raise (Exception('No nonbonded force found'))


def apply_radii_to_GB_force(radii, gb_force):
    """Given an array of radii, of length = gb_force.getNumParameters(),
    overwrite the per-particle radii parameters of gb_force.
    Retain charges, set scalingFactors to 1.0.
    """
    # TODO: Don't reset scalingFactor

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
