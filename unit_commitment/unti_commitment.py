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
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, diag, concatenate, append, matlib
import matplotlib.pyplot as plt


def main(case):
    """
    :param case: The test case for unit commitment problem
    :return:
    """
    from unit_commitment.data_format.data_format import IG, PG
    from unit_commitment.test_cases.case118 import F_BUS, T_BUS, BR_X, RATE_A
    from unit_commitment.test_cases.case118 import GEN_BUS, COST_C, COST_B, COST_A, PG_MAX, PG_MIN, I0, MIN_DOWN, \
        MIN_UP, RU, RD, COLD_START
    from unit_commitment.test_cases.case118 import BUS_ID, PD
    baseMVA, bus, gen, branch, profile = case["baseMVA"], case["bus"], case["gen"], case["branch"], case["Load_profile"]

    # Modify the bus, gen and branch matrix
    bus[:, BUS_ID] = bus[:, BUS_ID] - 1
    gen[:, GEN_BUS] = gen[:, GEN_BUS] - 1
    branch[:, F_BUS] = branch[:, F_BUS] - 1
    branch[:, T_BUS] = branch[:, T_BUS] - 1

    nb = shape(case['bus'])[0]  # number of buses
    nl = shape(case['branch'])[0]  # number of branches
    ng = shape(case['gen'])[0]  # number of schedule injections
    # Formulate a mixed integer quadratic programming problem
    # 1) Announce the variables
    # 1.1) boundary information
    T = case["Load_profile"].shape[0]
    lb = append(zeros((ng, 1)), gen[:, PG_MIN])
    ub = append(ones((ng, 1)), gen[:, PG_MAX])
    LB = matlib.repmat(lb, 1, T)
    UB = matlib.repmat(ub, 1, T)
    nx = LB.size
    # 1.2) boundary information
    vtypes = []
    for i in range(T):
        vtypes += ["B"] * ng
        vtypes += ["C"] * ng
    # 1.3) objective information
    c = append(gen[:, COST_C], gen[:, COST_B])
    C = matlib.repmat(c, 1, T)
    q = append(zeros((ng, 1)), gen[:, COST_A])
    Q = matlib.repmat(q, 1, T)
    Q = diag(Q[0])
    # 2) Constraint set
    # 2.1) Power balance equation
    Aeq = zeros((T, nx))
    for i in range(T):
        Aeq[i, i * 2 * ng + ng:(i + 1) * 2 * ng] = 1
    beq = zeros((T, 1))
    for i in range(T):
        beq[i] = case["Load_profile"][i]
    plt.plot(LB[0])
    plt.show()
    model["c"] = C
    
    return model


if __name__ == "__main__":
    from unit_commitment.test_cases import case118

    test_case = case118.case118()
    model = main(test_case)
