import numpy as np


mbondi_radius_dict = {}

# a few elements have different radii in the default model
# https://github.com/pandegroup/openmm/blob/master/wrappers/python/simtk/openmm/app/internal/customgbforces.py#L233-L257
mbondi_radius_dict[1] = 0.12
mbondi_radius_dict[6] = 0.17
mbondi_radius_dict[7] = 0.155
mbondi_radius_dict[8] = 0.15
mbondi_radius_dict[9] = 0.15
mbondi_radius_dict[14] = 0.21
mbondi_radius_dict[15] = 0.185
mbondi_radius_dict[16] = 0.18
# and default radii are 0.15

# scales
mbondi_scale_dict = {}

# https://github.com/pandegroup/openmm/blob/master/wrappers/python/simtk/openmm/app/internal/customgbforces.py#L384-L393
mbondi_scale_dict[1] = 0.85
mbondi_scale_dict[6] = 0.72
mbondi_scale_dict[7] = 0.79
mbondi_scale_dict[8] = 0.85
mbondi_scale_dict[9] = 0.88
mbondi_scale_dict[14] = 0.8
mbondi_scale_dict[15] = 0.86
mbondi_scale_dict[16] = 0.96
# and default scale factors are 0.8

element_numbers = sorted(list(set(mbondi_scale_dict.keys())))

from bayes_implicit_solvent.typers import GBTypingTree

from simtk import unit
mbondi_model = GBTypingTree(default_parameters={'radius' : 0.15 * unit.nanometer,
                                              'scale_factor': 0.8,
                                                },
                            proposal_sigmas={'radius' : 0.01 * unit.nanometer,
                                                                      'scale_factor': 0.01,
                                                                      }
                            )

from simtk import unit
mbondi_model.set_radius('*', 0.15 * unit.nanometer)
mbondi_model.set_scale_factor('*', 0.8)

for element_number in element_numbers:
    smirks = '[#{}]'.format(element_number)
    mbondi_model.add_child(child_smirks=smirks, parent_smirks='*')
    mbondi_model.set_radius(smirks, mbondi_radius_dict[element_number] * unit.nanometer)
    mbondi_model.set_scale_factor(smirks, mbondi_scale_dict[element_number])
mbondi_model.update_node_order()
mbondi_model.un_delete_able_types = set(mbondi_model.nodes)


# mbondi2 model
# % atomtype   radius (A)   scalingFactor
# H            1.20         0.85
# HN           1.30         0.85
# C            1.70         0.72
# N            1.55         0.79
# O            1.50         0.85
# F            1.50         0.88
# Si           2.10         0.80
# P            1.85         0.86
# S            1.80         0.96
# Cl           1.70         0.80
# Br           1.50         0.80
# I            1.50         0.80

# Cloned:
# *                (r = 1.50 Å, s = 0.80)
# |-[#1]           (r = 1.20 Å, s = 0.85)
#   |-[#1]~[#7]    (r = 1.30 Å, s = 0.85)
# |-[#6]           (r = 1.70 Å, s = 0.72)
# |-[#7]           (r = 1.55 Å, s = 0.79)
# |-[#8]           (r = 1.50 Å, s = 0.85)
# |-[#9]           (r = 1.50 Å, s = 0.88)
# |-[#14]          (r = 2.10 Å, s = 0.80)
# |-[#15]          (r = 1.85 Å, s = 0.86)
# |-[#16]          (r = 1.80 Å, s = 0.96)
# |-[#17]          (r = 1.70 Å, s = 0.80)
# |-[#35]          (r = 1.50 Å, s = 0.80)
# |-[#53]          (r = 1.50 Å, s = 0.80)

mbondi2_radius_dict = {}
mbondi2_radius_dict['[#1]'] = 0.12
mbondi2_radius_dict['[#1]~[#7]'] = 0.13
mbondi2_radius_dict['[#6]'] = 0.17
mbondi2_radius_dict['[#7]'] = 0.155
mbondi2_radius_dict['[#8]'] = 0.15
mbondi2_radius_dict['[#9]'] = 0.15
mbondi2_radius_dict['[#14]'] = 0.21
mbondi2_radius_dict['[#15]'] = 0.185
mbondi2_radius_dict['[#16]'] = 0.18
mbondi2_radius_dict['[#17]'] = 0.17
mbondi2_radius_dict['[#35]'] = 0.15
mbondi2_radius_dict['[#53]'] = 0.15

mbondi2_scale_dict = {}
mbondi2_scale_dict['[#1]'] = 0.85
mbondi2_scale_dict['[#1]~[#7]'] = 0.85
mbondi2_scale_dict['[#6]'] = 0.72
mbondi2_scale_dict['[#7]'] = 0.79
mbondi2_scale_dict['[#8]'] = 0.85
mbondi2_scale_dict['[#9]'] = 0.88
mbondi2_scale_dict['[#14]'] = 0.80
mbondi2_scale_dict['[#15]'] = 0.86
mbondi2_scale_dict['[#16]'] = 0.96
mbondi2_scale_dict['[#17]'] = 0.80
mbondi2_scale_dict['[#35]'] = 0.80
mbondi2_scale_dict['[#53]'] = 0.80


mbondi2_model = GBTypingTree(default_parameters={'radius' : 0.15 * unit.nanometer,
                                              'scale_factor': 0.8,
                                                },
                            proposal_sigmas={'radius' : 0.01 * unit.nanometer,
                                                                      'scale_factor': 0.01,
                                  }
                            )

mbondi2_model.set_radius('*', 0.15 * unit.nanometer)
mbondi2_model.set_scale_factor('*', 0.8)

for smirks in mbondi2_radius_dict.keys():
    if smirks == '[#1]~[#7]':
        parent_smirks = '[#1]'
    else:
        parent_smirks = '*'
    mbondi2_model.add_child(child_smirks=smirks, parent_smirks=parent_smirks)
    mbondi2_model.set_radius(smirks, mbondi2_radius_dict[smirks] * unit.nanometer)
    mbondi2_model.set_scale_factor(smirks, mbondi2_scale_dict[smirks])
mbondi2_model.update_node_order()
mbondi2_model.un_delete_able_types = set(mbondi_model.nodes)

if __name__ == '__main__':

    print('mbondi model: ')
    print(mbondi_model)
    print(mbondi_model.get_radii())
    print(mbondi_model.get_scale_factors())

    print('\n\nmbondi2 model: ')

    print(mbondi2_model)
    print(mbondi2_model.get_radii())
    print(mbondi2_model.get_scale_factors())
