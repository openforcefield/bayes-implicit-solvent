from openmmtools.constants import kB
from simtk import unit

temperature = 298.15 * unit.kelvin
beta = 1.0 / (kB * temperature)
RADIUS_UNIT = unit.nanometer


min_r, max_r = 0.01, 1.0
# TODO: sort out unit
min_scale, max_scale = 0.01, 10.0
# TODO: double-check this
