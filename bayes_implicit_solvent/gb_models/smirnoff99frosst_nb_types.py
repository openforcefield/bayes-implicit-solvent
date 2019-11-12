from bayes_implicit_solvent.typers import GBTypingTree

from simtk import unit
typer = GBTypingTree(default_parameters={'radius' : 0.15 * unit.nanometer,
                                              'scale_factor': 0.8,
                                              },
                        proposal_sigmas={'radius' : 0.01 * unit.nanometer,
                                                                      'scale_factor': 0.01,
                                                                      }
                          )

# hydrogen types
typer.add_child('[#1]', '*')
typer.add_child('[#1]-[#6X4]', '[#1]')
typer.add_child('[#1]-[#6X4]-[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X4]')
typer.add_child('[#1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]',
               '[#1]-[#6X4]-[#7,#8,#9,#16,#17,#35]')
typer.add_child('[#1]-[#6X4]~[*+1,*+2]', '[#1]-[#6X4]')
typer.add_child('[#1]-[#6X3]', '*')
typer.add_child('[#1]-[#6X3]~[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X3]')
typer.add_child('[#1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X3]~[#7,#8,#9,#16,#17,#35]')
typer.add_child('[#1]-[#6X2]', '[#1]')
typer.add_child('[#1]-[#7]', '[#1]')
typer.add_child('[#1]-[#8]', '[#1]')
typer.add_child('[#1]-[#16]', '[#1]')

# carbon types
typer.add_child('[#6]', '*')
typer.add_child('[#6X2]', '[#6]')
typer.add_child('[#6X4]', '[#6]')

# nitrogen type
typer.add_child('[#7]', '*')

# oxygen types
typer.add_child('[#8]', '*')
typer.add_child('[#8X2H0+0]', '[#8]')
typer.add_child('[#8X2H1+0]', '[#8]')

# fluorine types
typer.add_child('[#9]', '*')

# phosphorus type
typer.add_child('[#15]', '*')

# sulfur type
typer.add_child('[#16]', '*')

# chlorine type
typer.add_child('[#17]', '*')

# bromine type
typer.add_child('[#35]', '*')

# iodine type
typer.add_child('[#53]', '*')

if __name__ == '__main__':
    print(typer.number_of_nodes)