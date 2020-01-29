# openff imports
import openforcefield
from openforcefield.topology import Molecule
from openforcefield.typing.engines.smirnoff import ForceField
forcefield = ForceField('openff-1.0.0.offxml')
print(openforcefield._version.get_versions())


# extract smirks from forcefield
import re
def extract_all_smirks(forcefield):
    prefix = 'smirks="'
    matches = re.findall(r'smirks=.*"', forcefield.to_string())
    return [m.split()[0][len(prefix):-1] for m in matches]

def extract_all_vdW_smirks(forcefield):
    ff_string = forcefield.to_string()
    start_ind = ff_string.find('<vdW')
    end_ind = ff_string.find('</vdW')
    
    prefix = 'smirks="'
    matches = re.findall(r'smirks=.*"', ff_string[start_ind:end_ind])
    return [m.split()[0][len(prefix):-1] for m in matches]

all_smirks = extract_all_smirks(forcefield)
all_vdW_smirks = extract_all_vdW_smirks(forcefield)

# parse
import lark
with open('basic_smirks.lark', 'r') as f:
    grammar = lark.Lark(f)

parses = []

for s in all_smirks:
    try:
        parses.append(grammar.parse(s))
    except lark.exceptions.UnexpectedCharacters as e:
        parses.append(False)
        print('\n'.join(str(e).split('\n')[2:4]))

n_success = sum([type(p) == lark.tree.Tree for p in parses])
n_total = len(parses)
print('successfully parsed {} of {} smirks patterns! ({:.3f}%)'.format(n_success, n_total, 100 * n_success/n_total))
