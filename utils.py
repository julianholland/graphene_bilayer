
from ase.spacegroup.symmetrize import check_symmetry

def assign_point_group(structure):
    sym_dict={
        'C1':[1],
        'Ci':[2],
        'C2':list(range(3,6)),
        'Cs':list(range(6,10)),
        'C2h':list(range(10,16)),
        'D2':list(range(16,25)),
        'C2v':list(range(25,47)),
        'D2h':list(range(47,75)),
        'C4':list(range(75,81)),
        'S4':list(range(81,83)),
        'C4h':list(range(83,89)),
        'D4':list(range(89,99)),
        'C4v':list(range(99,111)),
        'D2d':list(range(111,123)),
        'D4h':list(range(123,143)),
        'C3':list(range(143,147)),
        'S6':list(range(147,149)),
        'D3':list(range(149,156)),
        'C3v':list(range(156,162)),
        'D3d':list(range(162,168)),
        'C6':list(range(168,174)),
        'C3h':list(range(174,175)),
        'C6h':list(range(175,177)),
        'D6':list(range(177,183)),
        'C6v':list(range(183,187)),
        'D3h':list(range(187,191)),
        'D6h':list(range(191,195)),
        'T':list(range(195,200)),
        'Th':list(range(200,207)),
        'O':list(range(207,215)),
        'Td':list(range(215,221)),
        'Oh':list(range(221,231)),
    }

    sym_number=(check_symmetry(structure)['number'])
    for key in sym_dict:
        if sym_number in sym_dict[key]:
            point_group=key
    return point_group     