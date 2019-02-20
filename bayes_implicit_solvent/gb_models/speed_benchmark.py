# ensure consistency between energies from OpenMM and from autograd / numpy clone

from autograd import numpy as np
from autograd import grad
from openmmtools.testsystems import AlanineDipeptideImplicit
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app

from bayes_implicit_solvent.gb_models.numpy_gb_models import compute_OBC_energy_reference, compute_OBC_energy_vectorized


testsystem = AlanineDipeptideImplicit()

gbsa_force = testsystem.system.getForce(4)

print('COMPUTED VALUES\n\n')
for i in range(gbsa_force.getNumComputedValues()):
    variable, expression, computation_type = gbsa_force.getComputedValueParameters(i)
    expressions = expression.split(';')
    print('computation type: {}'.format(computation_type))
    print('{} = {}'.format(variable, expressions[0]))
    for e in expressions[1:]:
        print('\t{}'.format(e.strip()))
    print()

print('ENERGY TERMS\n\n')
for i in range(gbsa_force.getNumEnergyTerms()):
    expression, computation_type = gbsa_force.getEnergyTermParameters(i)
    expressions = expression.split(';')
    print('computation type: {}'.format(computation_type))
    print(expressions[0])
    for e in expressions[1:]:
        print('\t{}'.format(e.strip()))
    print()

# quick test that the clone is doing the same as the original
implicit_sim = app.Simulation(testsystem.topology,
                              testsystem.system,
                              mm.LangevinIntegrator(298 * unit.kelvin, 1.0 / unit.picosecond,
                                                    1.0 * unit.femtosecond),
                              platform=mm.Platform.getPlatformByName('CPU')
                              )
implicit_sim.context.setPositions(testsystem.positions)
implicit_sim.step(1)

from tqdm import tqdm

print('collecting a few configurations...')
implicit_traj = []
for _ in tqdm(range(100)):
    implicit_sim.step(100)
    implicit_traj.append(implicit_sim.context.getState(getPositions=True).getPositions(asNumpy=True))

from scipy.spatial.distance import pdist, squareform

offset = 0.009
screening = 138.935456
surface_tension = 28.3919551
solvent_dielectric = 78.5
solute_dielectric = 1.0

implicit_only = AlanineDipeptideImplicit()
implicit_only.system.removeForce(0)
implicit_only.system.removeForce(0)
implicit_only.system.removeForce(0)
implicit_only.system.removeForce(0)
implicit_only.system.removeForce(1)
print(implicit_only.system.getForces())

implicit_only_sim = app.Simulation(testsystem.topology,
                                   implicit_only.system,
                                   mm.LangevinIntegrator(298 * unit.kelvin, 1.0 / unit.picosecond,
                                                         1.0 * unit.femtosecond),
                                   platform=mm.Platform.getPlatformByName('CPU')
                                   )
implicit_only_sim.context.setPositions(testsystem.positions)


timings = {}
from time import time

t0 = time()
implicit_only_Us = []
n_snapshots = len(implicit_traj)
for i in range(n_snapshots):
    implicit_only_sim.context.setPositions(implicit_traj[i])
    implicit_only_Us.append(implicit_only_sim.context.getState(getEnergy=True).getPotentialEnergy())
t1 = time()

reference_timing = (t1 - t0) / n_snapshots

print('cloning per particle parameters')

charges = []
o_rs = []
s_rs = []

for i in range(gbsa_force.getNumPerParticleParameters()):
    print(gbsa_force.getPerParticleParameterName(i))

for i in range(gbsa_force.getNumParticles()):
    charge, o_r, s_r = gbsa_force.getParticleParameters(i)
    charges.append(charge)
    o_rs.append(o_r)
    s_rs.append(s_r)
charges = np.array(charges)
radii = np.zeros(len(o_rs))
for i in range(len(o_rs)):
    radii[i] = o_rs[i] + offset
scales = np.zeros(len(s_rs))
for i in range(len(o_rs)):
    scales[i] = s_rs[i] / o_rs[i]

def precompute_distance_matrices(x):
    return [squareform(pdist(snapshot)) + np.eye(len(snapshot)) for snapshot in x]

distance_matrices = precompute_distance_matrices(implicit_traj)


from time import time

t0 = time()
autograd_reference_energies = [compute_OBC_energy_reference(distance_matrix, radii, scales, charges,
                                                            offset, screening, surface_tension,
                                                            solvent_dielectric, solute_dielectric) for
                               distance_matrix in

                               distance_matrices]
t1 = time()
timings['Autograd reference energies'] = (t1 - t0) / n_snapshots


t0 = time()
autograd_vectorized_energies = [compute_OBC_energy_vectorized(distance_matrix, radii, scales, charges,
                                                              offset, screening, surface_tension,
                                                              solvent_dielectric, solute_dielectric) for
                                distance_matrix in
                                distance_matrices]
t1 = time()
timings['Autograd vectorized energies'] = (t1 - t0) / n_snapshots


def pack(radii, scales):
    n = len(radii)
    theta = np.zeros(2 * n)
    theta[:n] = radii
    theta[n:2 * n] = scales
    return theta

def unpack(theta):
    n = int((len(theta)) / 2)
    radii, scales = theta[:n], theta[n:2 * n]
    return radii, scales

def reference_loss(theta):
    radii, scales = unpack(theta)
    energies = [compute_OBC_energy_reference(distance_matrix, radii, scales, charges,
                                   offset, screening, surface_tension,
                                   solvent_dielectric, solute_dielectric) for
     distance_matrix in
     distance_matrices]
    loss = np.sum(np.abs(energies))
    return loss

def vectorized_loss(theta):
    radii, scales = unpack(theta)
    energies = [compute_OBC_energy_vectorized(distance_matrix, radii, scales, charges,
                                   offset, screening, surface_tension,
                                   solvent_dielectric, solute_dielectric) for
     distance_matrix in
     distance_matrices]
    loss = np.sum(np.abs(energies))
    return loss


theta = pack(radii, scales)

t0 = time()
_ = grad(vectorized_loss)(theta)
t1 = time()
timings['Autograd vectorized parameter gradients'] = (t1 - t0) / n_snapshots

# somehow this gets a "ValueError: setting an array element with a sequence"
#t0 = time()
#_ = grad(reference_loss)(theta)
#t1 = time()
#timings['Autograd reference parameter gradients'] = (t1 - t0) / n_snapshots

from autograd import hessian_vector_product
vector = np.random.randn(len(theta))

t0 = time()
_ = hessian_vector_product(vectorized_loss)(theta, vector)
t1 = time()
timings['Autograd vectorized parameter hessian-vector products'] = (t1 - t0) / n_snapshots


print('OpenMM reference timing: {}s per snapshot'.format(reference_timing))
for key in timings:
    print('{}: {:.3f}s ({:.3f}x OpenMM reference)'.format(key, timings[key], timings[key] / reference_timing))
