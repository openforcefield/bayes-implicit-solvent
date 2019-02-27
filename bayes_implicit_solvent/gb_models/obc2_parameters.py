import numpy as np


initial_radius_dict = {}

# a few elements have different radii in the default model
# https://github.com/pandegroup/openmm/blob/master/wrappers/python/simtk/openmm/app/internal/customgbforces.py#L233-L257
initial_radius_dict[1] = 0.12
initial_radius_dict[6] = 0.17
initial_radius_dict[7] = 0.155
initial_radius_dict[8] = 0.15
initial_radius_dict[9] = 0.15
initial_radius_dict[14] = 0.21
initial_radius_dict[15] = 0.185
initial_radius_dict[16] = 0.18
# and default radii are 0.15

# scales
initial_scale_dict = {}

# a few elements have different radii in the default model
# https://github.com/pandegroup/openmm/blob/master/wrappers/python/simtk/openmm/app/internal/customgbforces.py#L233-L257
initial_scale_dict[1] = 0.85
initial_scale_dict[6] = 0.72
initial_scale_dict[7] = 0.79
initial_scale_dict[8] = 0.85
initial_scale_dict[9] = 0.88
initial_scale_dict[14] = 0.8
initial_scale_dict[15] = 0.86
initial_scale_dict[16] = 0.96
# and default scale factors are 0.8

element_numbers = sorted(list(set(initial_scale_dict.keys())))

from bayes_implicit_solvent.typers import GBTypingTree

from simtk import unit
obc2_model = GBTypingTree(default_parameters={'radius' : 0.15 * unit.nanometer,
                                              'scale_factor': 0.8,
                                              },
                        proposal_sigmas={'radius' : 0.01 * unit.nanometer,
                                                                      'scale_factor': 0.01,
                                                                      }
                          )

from simtk import unit
obc2_model.set_radius('*', 0.15 * unit.nanometer)
obc2_model.set_scale_factor('*', 0.8)

for element_number in element_numbers:
    smirks = '[#{}]'.format(element_number)
    obc2_model.add_child(child_smirks=smirks, parent_smirks='*')
    obc2_model.set_radius(smirks, initial_radius_dict[element_number] * unit.nanometer)
    obc2_model.set_scale_factor(smirks, initial_scale_dict[element_number])
obc2_model.update_node_order()
obc2_model.un_delete_able_types = set(obc2_model.nodes)

if __name__ == '__main__':
    print(obc2_model)
    print(obc2_model.get_radii())
    print(obc2_model.get_scale_factors())
