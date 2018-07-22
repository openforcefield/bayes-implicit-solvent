import os

path_to_base_forcefield = "forcefields/smirnoff99Frosst.ffxml"

def truncated_ff_factory(path_to_base_forcefield, n_types=1):
    """Truncates the number of NonbondedForce SMIRKS "types" to n_types"""

    # modify the Nonbonded XML definition, create a new force-field

    with open(path_to_base_forcefield, "r") as f:
        lines = f.readlines()
    i_start = [i for i in range(len(lines)) if "<NonbondedForce" in lines[i]][0] + 1
    i_end = [i for i in range(len(lines)) if "</NonbondedForce>" in lines[i]][0]
    nonbonded_type_lines = lines[i_start:i_end]

    # un-modified beginning of file
    new_lines = []
    for i in range(i_start):
        new_lines.append(lines[i])

    # truncated section of file in NonbondedForce definition
    for i in range(n_types):
        new_lines.append(nonbonded_type_lines[i])

    # un-modified end of file
    for i in range(i_end, len(lines)):
        new_lines.append(lines[i])

    dirname, basename = os.path.split(path_to_base_forcefield)

    name, extension = basename.split('.')
    new_forcefield_name = name + '_{}.'.format(n_types) + extension

    path_to_new_forcefield = os.path.join(dirname, new_forcefield_name)
    with open(path_to_new_forcefield, 'w') as f:
        f.writelines(new_lines)

    return path_to_new_forcefield

for i in range(1, 36):
    print(truncated_ff_factory(path_to_base_forcefield, n_types=i))
