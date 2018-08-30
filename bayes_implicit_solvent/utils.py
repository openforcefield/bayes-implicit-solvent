from openeye import oechem
from simtk import unit


def smiles_to_mol(smiles):
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smiles)
    return mol


def get_gbsa_force(system):
    for force in system.getForces():
        if "GBSA" in force.__class__.__name__:
            return force
    raise (Exception('No GBSA force found'))

def get_nb_force(system):
    forces = system.getForces()
    for f in forces:
        if 'NonbondedForce' in f.__class__.__name__:
            return f
    raise (Exception('No nonbonded force found'))

def apply_radii_to_GB_force(radii, gb_force):
    """Given an array of radii, of length = gb_force.getNumParameters(),
    overwrite the radii parameters
    """

    for i in range(len(radii)):
        charge = gb_force.getParticleParameters(i)[0]
        gb_force.setParticleParameters(index=i,
                                   charge=charge,
                                   radius=radii[i],
                                   scalingFactor=1.0,
                                   )


def mdtraj_to_list_of_unitted_snapshots(traj):
    return [snapshot * unit.nanometer for snapshot in traj.xyz]