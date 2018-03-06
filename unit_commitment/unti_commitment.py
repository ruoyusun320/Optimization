"""
Basic unit commitment to some mix-integer linear/quadratic programming problem
@author: Zhao Tianyang
@e-mail: zhaoty@ntu.edu.sg
@date:6 Mar 2018
"""
from pypower.loadcase import loadcase
from pypower.ext2int import ext2int
from pypower.ext2int import ext2int
from numpy import flatnonzero as find
from scipy.sparse.linalg import inv
from scipy.sparse import vstack
from numpy import zeros, c_, shape, ix_,ones,r_,arange,sum,diag,concatenate

from pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I
from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN


def main(case):
    """
    :param case: The test case for unit commitment problem
    :return:
    """
    from unit_commitment.data_format import IG, PG
    case = loadcase(case)
    case = ext2int(case)
    baseMVA, bus, gen, branch, gencost = case["baseMVA"], case["bus"], case["gen"], case["branch"], case["gencost"]  #
    nb = shape(case['bus'])[0]  # number of buses
    nl = shape(case['branch'])[0]  # number of branches
    ng = shape(case['gen'])[0]  # number of schedule injections


    return model


if __name__ == "__main__":
    from pypower import case118

    test_case = case118.case118()
    model = main(test_case)
