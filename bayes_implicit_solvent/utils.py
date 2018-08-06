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

