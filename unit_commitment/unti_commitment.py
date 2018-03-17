"""
Basic unit commitment to some mix-integer linear/quadratic programming problem
@author: Zhao Tianyang
@e-mail: zhaoty@ntu.edu.sg
@date:6 Mar 2018
"""
from pypower.loadcase import loadcase
from pypower.ext2int import ext2int
from numpy import flatnonzero as find
from scipy.sparse.linalg import inv
from scipy.sparse import vstack
from numpy import zeros, c_, shape, ix_,ones,r_,arange,sum,diag,concatenate



def main(case):
    """
    :param case: The test case for unit commitment problem
    :return:
    """
    from unit_commitment.data_format.data_format import IG, PG

    baseMVA, bus, gen, branch, profile = case["baseMVA"], case["bus"], case["gen"], case["branch"],case["Load_profile"]
    nb = shape(case['bus'])[0]  # number of buses
    nl = shape(case['branch'])[0]  # number of branches
    ng = shape(case['gen'])[0]  # number of schedule injections


    return model


if __name__ == "__main__":
    from unit_commitment.test_cases import case118

    test_case = case118.case118()
    model = main(test_case)
