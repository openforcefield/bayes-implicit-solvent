atomic_number_dict = {}
for i in [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]:
    atomic_number_dict['[#{}]'.format(i)] = 'atomic number = {}'.format(i)


def construct_atomic_primitives_dict(
        #max_degree=4,  # maximum from FreeSolv
        max_attached_hydrogens=4,  # maximum from FreeSolv
        max_implicit_hydrogens=0,  # maximum from FreeSolv
        max_ring_membership=4,
        max_ring_size=6,
        max_valence=6,  # maximum from FreeSolv
        max_connectivity=4,
        max_ring_connectivity=4,
        min_negative_charge=-1,  # minimum from FreeSolv
        max_positive_charge=2,  # maximum from FreeSolv
        all_atomic_nums=[1, 6, 7, 8, 9, 15, 16, 17, 35, 53],  # set of elements in FreeSolv
):
    """Construct a dictionary with SMARTS symbols for atomic primitives as keys,
    and human-readable descriptions as values.

    Ripped off from the table in section 4.1 of
    http://www.daylight.com/dayhtml/doc/theory/theory.smarts.html

    TODO: Potentially also include chirality properties
    """

    # initialize dictionary
    atomic_primitives = {
        '*': 'wildcard',
        'a': 'aromatic',
        'A': 'aliphatic',
    }

    # OpenEye can't parse this...
    ## add degree properties
    #for i in range(1, max_degree + 1):
    #    atomic_primitives['D{}'.format(i)] = 'degree = {} ({} explicit connections)'.format(i, i)

    # add attached hydrogen properties
    for i in range(0, max_attached_hydrogens + 1):
        atomic_primitives['H{}'.format(i)] = 'total-H-count = {} ({} attached hydrogens)'.format(i, i)

    # add implicit hydrogen properties
    for i in range(0, max_implicit_hydrogens + 1):
        atomic_primitives['h{}'.format(i)] = 'implicit-H-count = {} ({} implicit hydrogens)'.format(i, i)

    # add ring membership properties
    for i in range(0, max_ring_membership + 1):
        atomic_primitives['R{}'.format(i)] = 'ring membership = {} (in {} SSSR rings)'.format(i, i)

    # add ring size properties
    for i in range(0, max_ring_size + 1):
        atomic_primitives['r{}'.format(i)] = 'ring size = {} (in smallest SSSR ring of size {})'.format(i, i)

    # add valence properties
    for i in range(1, max_valence + 1):
        atomic_primitives['v{}'.format(i)] = 'valence = {} (total bond order {})'.format(i, i)

    # add connectivity properties
    for i in range(1, max_connectivity + 1):
        atomic_primitives['X{}'.format(i)] = 'connectivity = {} ({} total connections)'.format(i, i)

    # add ring connectivity properties
    for i in range(0, max_ring_connectivity + 1):
        atomic_primitives['x{}'.format(i)] = 'ring connectivity = {} ({} total ring connections)'.format(i, i)

    # add charge properties
    for i in range(min_negative_charge, 0):
        atomic_primitives[str(i)] = 'negative charge = {} ({} charge)'.format(i, i)
    for i in range(1, max_positive_charge + 1):
        atomic_primitives['+{}'.format(i)] = 'positive charge = {} (+{} formal charge)'.format(i, i)

    # add atomic number properties
    for i in all_atomic_nums:
        atomic_primitives['#{}'.format(i)] = 'atomic number = {}'.format(i)

    return atomic_primitives

atomic_primitives = construct_atomic_primitives_dict()

unary_logical_operators = {
    '!': 'not',
}

binary_logical_operators = {
    '&': 'and (high precedence)',
    ',': 'or',
    ';': 'and (low precedence)',
}