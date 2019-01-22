from openmmtools.constants import kB
from simtk import unit

temperature = 298.15 * unit.kelvin
beta = 1.0 / (kB * temperature)
RADIUS_UNIT = unit.nanometer
