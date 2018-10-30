from bayes_implicit_solvent.typers import GBTypingTree

typer = GBTypingTree()

# hydrogen types
typer.add_child('[#1]-[#6X4]', '*')
typer.add_child('[#1]-[#6X4]-[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X4]')
typer.add_child('[#1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X4]-[#7,#8,#9,#16,#17,#35]')
typer.add_child('[#1]-[#6X4]~[*+1,*+2]', '[#1]-[#6X4]')
typer.add_child('[#1]-[#6X3]', '*')
typer.add_child('[#1]-[#6X3]~[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X3]')
typer.add_child('[#1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X3]~[#7,#8,#9,#16,#17,#35]')
typer.add_child('[#1]-[#6X2]', '*')
typer.add_child('[#1]-[#7]', '*')
typer.add_child('[#1]-[#8]', '*')
typer.add_child('[#1]-[#16]', '*')

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

print(typer)

from bayes_implicit_solvent.prior_checking import check_no_empty_types
check_no_empty_types(typer)
