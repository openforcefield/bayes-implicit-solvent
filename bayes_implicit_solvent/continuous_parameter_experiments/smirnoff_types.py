from bayes_implicit_solvent.typers import GBTypingTree

typer = GBTypingTree()

# hydrogen types
typer.add_child('[#1]', '*')
typer.add_child('[#1]-[#6X4]', '[#1]')
typer.add_child('[#1]-[#6X4]-[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X4]')
typer.add_child('[#1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X4]-[#7,#8,#9,#16,#17,#35]')
typer.add_child('[#1]-[#6X4]~[*+1,*+2]', '[#1]-[#6X4]')
typer.add_child('[#1]-[#6X3]', '[#1]')
typer.add_child('[#1]-[#6X3]~[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X3]')
typer.add_child('[#1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X3]~[#7,#8,#9,#16,#17,#35]')
typer.add_child('[#1]-[#6X2]', '[#1]')
typer.add_child('[#1]-[#7]', '[#1]')
typer.add_child('[#1]-[#8]', '[#1]')
typer.add_child('[#1]-[#16]', '[#1]')

# lithium type
typer.add_child('[#3+1]', '*')

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
typer.add_child('[#9X0-1]', '[#9]')  # TODO: Remove?

# sodium
typer.add_child('[#11+1]', '*')

# phosphorus type
typer.add_child('[#15]', '*')

# sulfur type
typer.add_child('[#16]', '*')

# chlorine type
typer.add_child('[#17]', '*')
typer.add_child('[#17X0-1]', '[#17]')  # TODO: Remove?

# potassium type
typer.add_child('[#19+1]', '*')  # TODO: Remove?

# rubidium type
typer.add_child('[#37+1]', '*')  # TODO: Remove?

# cesium type
typer.add_child('[#55+1]', '*')  # TODO: Remove?

# bromine type
typer.add_child('[#35X0-1]', '*')  # TODO: Remove?

# iodine type
typer.add_child('[#53X0-1]', '*')   # TODO: Remove?

print(typer)
