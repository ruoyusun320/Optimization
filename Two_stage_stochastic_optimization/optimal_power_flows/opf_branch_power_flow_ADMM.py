"""
ADMM based distributed optimal power flow
The power flow modelling is based on the branch power flow

References:
    [1]Peng, Qiuyu, and Steven H. Low. "Distributed optimal power flow algorithm for radial networks, I: Balanced single phase case." IEEE Transactions on Smart Grid (2016).
    The full model can be applied, with the implement of Algorithm 2.
"""

from Two_stage_stochastic_optimization.power_flow_modelling import case33
from gurobipy import *
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, diag, concatenate, where, inf
from scipy.sparse import csr_matrix as sparse

# Data format
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I, QD
from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int


def run(mpc):
    """
    Gurobi based optimal power flow modelling and solution
    :param mpc: The input case of optimal power flow
    :return: obtained solution
    """
    mpc = ext2int(mpc)
    baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

    nb = shape(mpc['bus'])[0]  # number of buses
    nl = shape(mpc['branch'])[0]  # number of branches
    ng = shape(mpc['gen'])[0]  # number of dispatchable injections
    f = branch[:, F_BUS]  ## list of "from" buses
    t = branch[:, T_BUS]  ## list of "to" buses

    # Modify the bus information
    Branch_R = branch[:, BR_R]
    Branch_X = branch[:, BR_X]
    Slmax = branch[:, RATE_A] / baseMVA
    gen[:, PMAX] = gen[:, PMAX] / baseMVA
    gen[:, PMIN] = gen[:, PMIN] / baseMVA
    gen[:, QMAX] = gen[:, QMAX] / baseMVA
    gen[:, QMIN] = gen[:, QMIN] / baseMVA
    gencost[:, 4] = gencost[:, 4] * baseMVA * baseMVA
    gencost[:, 5] = gencost[:, 5] * baseMVA
    bus[:, PD] = bus[:, PD] / baseMVA
    bus[:, QD] = bus[:, QD] / baseMVA

    area = ancestor_children_generation(f, t, nb, Branch_R, Branch_X, Slmax, gen, bus, gencost, baseMVA)
    M = inf


def turn_to_power(list, power=1):
    return [number ** power for number in list]


def ancestor_children_generation(branch_f, branch_t, nb, Branch_R, Branch_X, SMAX, gen, bus, gencost, baseMVA):
    """
    Ancestor and children information for each node, together with information within each area
    :param branch_f:
    :param branch_t:
    :param index: Bus index
    :param Branch_R: Branch resistance
    :param Branch_X: Branch reactance
    :param SMAX: Current limitation within each area
    :param gen: Generation information
    :param bus: Bus information
    :return: Area, ancestor bus, children buses, line among buses, load, generations and line information
    """
    Area = []
    for i in range(nb):
        temp = {}
        temp["Index"] = i
        if i in branch_t:
            AncestorBus = branch_f[where(branch_t == i)]
            temp["Ai"] = int(AncestorBus[0])  # For each bus, there exits only one ancestor bus, as one connected tree
            AncestorBranch = where(branch_t == i)
            temp["Abranch"] = int(AncestorBranch[0])
            temp["BR_R"] = Branch_R[temp["Abranch"]]
            temp["BR_X"] = Branch_X[temp["Abranch"]]
            temp["SMAX"] = SMAX[temp["Abranch"]]
            if i in branch_f:
                temp["TYPE"] = "RELAY"
            else:
                temp["TYPE"] = "LEAF"
        else:
            temp["Ai"] = []
            temp["Abranch"] = []
            temp["BR_R"] = 0
            temp["BR_X"] = 0
            temp["SMAX"] = 0
            temp["TYPE"] = "ROOT"

        if i in branch_f:
            ChildrenBranch = where(branch_f == i)
            nChildren = len(ChildrenBranch[0])
            temp["Ci"] = []
            temp["Cbranch"] = []
            for j in range(nChildren):
                temp["Cbranch"].append(int(ChildrenBranch[0][j]))
                temp["Ci"].append(int(branch_t[temp["Cbranch"][j]]))  # The children bus
        else:
            temp["Cbranch"] = []
            temp["Ci"] = []

        # Update the node information
        if i in gen[:, GEN_BUS]:
            temp["PGMAX"] = gen[where(gen[:, GEN_BUS] == i), PMAX][0][0]
            temp["PGMIN"] = gen[where(gen[:, GEN_BUS] == i), PMIN][0][0]
            temp["QGMAX"] = gen[where(gen[:, GEN_BUS] == i), QMAX][0][0]
            temp["QGMIN"] = gen[where(gen[:, GEN_BUS] == i), QMIN][0][0]
            if temp["PGMIN"] > temp["PGMAX"]:
                t = temp["PGMIN"]
                temp["PGMIN"] = temp["PGMAX"]
                temp["PGMAX"] = t
            if temp["QGMIN"] > temp["QGMAX"]:
                t = temp["QGMIN"]
                temp["QGMIN"] = temp["QGMAX"]
                temp["QGMAX"] = t
            temp["a"] = gencost[where(gen[:, GEN_BUS] == i), 4][0][0]
            temp["b"] = gencost[where(gen[:, GEN_BUS] == i), 5][0][0]
            temp["c"] = gencost[where(gen[:, GEN_BUS] == i), 6][0][0]
        else:
            temp["PGMAX"] = 0
            temp["PGMIN"] = 0
            temp["QGMAX"] = 0
            temp["QGMIN"] = 0
            temp["a"] = 0
            temp["b"] = 0
            temp["c"] = 0
        temp["PD"] = bus[i, PD]
        temp["QD"] = bus[i, QD]
        temp["VMIN"] = bus[i, VMIN] ** 2
        temp["VMAX"] = bus[i, VMAX] ** 2

        Area.append(temp)

    return Area


def sub_problem(Index, Area, ru):
    """
    Sub-problem optimization for each area, where the area is defined one bus and together with the
    :param Index: Target area
    :param Area: Area connection information
    :param ru: Penalty factor for the second order constraints
    :return: Area, updated information
    """

    modelX = Model("sub_opf_x")  # Sub-optimal power flow x update
    modelY = Model("sub_opf_y")  # Sub-optimal power flow y update

    if Area[Index]["Type"] == "ROOT":  # Only needs to meet the KCL equation
        # xi = [Pgi,Qgi,pi_x,qi_x,Vi_x]
        # zi= [pi_z,qi_z,Vi_z,Pj_i_z(j in children set of i),Qj_i_z,Ij_i_z]
        # The following types of constraints should be considered
        # 1) pi_x = pi_z
        # 2) qi_x = qi_z
        # 3) Vi_x = Vi_z
        # 4) Vi_x = V_A_j_z j in children set of i

        # 5）Pj_x = Pj_i_z j in children set of i, active power from j to i
        # 6）Qj_x = Qj_i_z j in children set of i
        # 7）Ij_x = Ij_i_z j in children set of i
        # constraint 5)-7) are managed by the children buses

        # Information exchange
        # In the z update, receive Pj_x, Qj_x, Ij_x and associate Lagrange multiper from children set
        # In the x update, receive V_A_j_z from children set

        # Step 1: construct the x update problem
        nChildren = len(Area[Index]["Ci"])
        V_Ci_z = []
        Pi_Ci_x = []
        Qi_Ci_x = []
        li_Ci_x = []
        mu_Pi_Ci_x = []
        mu_Qi_Ci_x = []
        mu_li_Ci_x = []
        Line_R = []
        Line_X = []
        for i in range(nChildren):  # Information exchange
            V_Ci_z.append(Area[Area[Index]["Ci"][i]]["V_Ai_z"])  # Voltage magnitude stored in children set

            Pi_Ci_x = Pi_Ci_x.append(
                Area[Area[Index]["Ci"][i]]["Pi_x"])  # Active power from i to Ai stored in children set
            Qi_Ci_x = Qi_Ci_x.append(
                Area[Area[Index]["Ci"][i]]["Qi_x"])  # Reactive power from i to Ai stored in children set
            li_Ci_x = li_Ci_x.append(Area[Area[Index]["Ci"][i]]["li_x"])  # Current from i to Ai stored in children set
            mu_Pi_Ci_x = mu_Pi_Ci_x.append(
                Area[Area[Index]["Ci"][i]]["mu_Pi_x"])  # Multiplier of active power from i to Ai stored in children set
            mu_Qi_Ci_x = mu_Qi_Ci_x.append(Area[Area[Index]["Ci"][i]][
                                               "mu_Qi_x"])  # Multiplier of reactive power from i to Ai stored in children set
            mu_li_Ci_x = mu_li_Ci_x.append(
                Area[Area[Index]["Ci"][i]]["mu_li_x"])  # Multiplier of current from i to Ai stored in children set
            Line_R = Line_R.append(Area[Area[Index]["Ci"][i]]["Line_R"])  # Line resistance of the children area
            Line_X = Line_X.append(Area[Area[Index]["Ci"][i]]["Line_X"])  # Line reactance of the children area

        if len(Area["Gen"]) != 0:
            pi_x = modelX.addVar(lb=Area[Index]["PMIN"] - Area[Index]["PD"], ub=Area[Index]["PMAX"] - Area[Index]["PD"],
                                 vtype=GRB.CONTINUOUS, name="pi")
            qi_x = modelX.addVar(lb=Area[Index]["QMIN"] - Area[Index]["QD"], ub=Area[Index]["QMAX"] - Area[Index]["QD"],
                                 vtype=GRB.CONTINUOUS, name="qi")
            Pg = modelX.addVar(lb=Area[Index]["PMIN"], ub=Area[Index]["PMAX"], vtype=GRB.CONTINUOUS, name="Pg")
            Qg = modelX.addVar(lb=Area[Index]["QMIN"], ub=Area[Index]["QMAX"], vtype=GRB.CONTINUOUS, name="Qg")

            pi_z = modelY.addVar(lb=Area[Index]["PMIN"] - Area[Index]["PD"], ub=Area[Index]["PMAX"] - Area[Index]["PD"],
                                 vtype=GRB.CONTINUOUS, name="pi")
            qi_z = modelY.addVar(lb=Area[Index]["QMIN"] - Area[Index]["QD"], ub=Area[Index]["QMAX"] - Area[Index]["QD"],
                                 vtype=GRB.CONTINUOUS, name="qi")

        else:
            pi_x = modelX.addVar(lb=- Area[Index]["PD"], ub=- Area[Index]["PD"], vtype=GRB.CONTINUOUS, name="Pi")
            qi_x = modelX.addVar(lb=- Area[Index]["QD"], ub=- Area[Index]["QD"], vtype=GRB.CONTINUOUS, name="Qi")
            Pg = modelX.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name="Pg")
            Qg = modelX.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name="Qg")
            pi_z = modelY.addVar(lb=- Area[Index]["PD"], ub=- Area[Index]["PD"], vtype=GRB.CONTINUOUS, name="Pi")
            qi_z = modelY.addVar(lb=- Area[Index]["QD"], ub=- Area[Index]["QD"], vtype=GRB.CONTINUOUS, name="Qi")

        Vi_x = modelX.addVar(lb=Area[Index]["VMIN"], ub=Area[Index]["VMAX"], vtype=GRB.CONTINUOUS, name="Vi")

        Pji_z = modelY.addVar(1, nChildren)
        Qji_z = modelY.addVar(1, nChildren)
        lji_z = modelY.addVar(1, nChildren)

        modelX.addConstr(Pg - pi_x == Area[Index]["PD"])
        modelX.addConstr(Qg - qi_x == Area[Index]["QD"])

        # KCL equations
        # 1) Active power balance equation
        expr = 0
        for i in range(nChildren):
            expr = expr + Pji_z[i] - lji_z[i] * Line_R[i]
        modelY.addConstr(lhs=expr + pi_z, sense=GRB.EQUAL, rhs=0)
        # 2) Reactive power balance equation
        expr = 0
        for i in range(nChildren):
            expr = expr + Qji_z[i] - lji_z[i] * Line_X[i]
        modelY.addConstr(lhs=expr + qi_z, sense=GRB.EQUAL, rhs=0)



    elif Area[Index]["Type"] == "LEAF":  # Only needs to meet the KVL equation
        # xi = [Pgi,Qgi,pi_x,qi_x,Vi_x,Ii_x,Pi_x,Qi_x]# Pi_x represent the power from i to its ancestor
        # zi = [qi_z,pi_z,Vi_z,Ii_z,Pi_z,Qi_z,V_A_i_z]#
        # The following types of constraints should be considered
        # 1) pi_x = pi_z
        # 2) qi_x = qi_z
        # 3) Vi_x = Vi_z
        # 4) Ii_x = Ii_z
        # 5) Pi_x = Pi_z
        # 6) Qi_x = Qi_z
        # 7) Pi_x = Pi_j_z j is the ancestor bus of i
        # 8) Qi_x = Qi_j_z j is the ancestor bus of i
        # 9) Ii_x = Ii_j_z j is the ancestor bus of i
        # 10) Vj_x = V_A_i_z  j is the ancestor bus of i

        # Information exchange
        # In the z update, receive Vj_x and associate Lagrange multiper from ancestor bus
        # In the x update, receive Pi_j_z, Qi_j_z and Ii_j_z from ancestor bus

        Pji_x = modelX.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name="Pij")

    else:  # Only needs to meet the KVL equation
        # xi = [Pgi,Qgi,pi_x,pi_x,Vi_x,Ii_x,Pi_x,Qi_x]# Pi_x represent the power from i to its ancestor
        # zi = [pi_z,pi_z,Vi_z,Ii_z,Pi_z,Qi_z,V_A_i_z,Pj_i_z(j in children set of i),Qj_i_z,Ij_i_z]#
        # The following types of constraints should be considered
        # 1) pi_x = pi_z
        # 2) qi_x = qi_z
        # 3) Vi_x = Vi_z
        # 4) Ii_x = Ii_z
        # 5) Pi_x = Pi_z
        # 6) Qi_x = Qi_z
        # 7) Pi_x = Pi_j_z j is the ancestor bus of i
        # 8) Qi_x = Qi_j_z j is the ancestor bus of i
        # 9) Ii_x = Ii_j_z j is the ancestor bus of i
        # 10) Vi_x = V_A_j_z  j in the children bus of i

        # 11) Vj_x = V_A_i_z j is the ancestor bus of i
        # 11）Pj_x = Pj_i_z j in children set of i
        # 12）Qj_x = Qj_i_z j in children set of i
        # 13）Ij_x = Ij_i_z j in children set of i

        # Information exchange
        # In the z update, receive Pj_x, Qj_x, Ij_x and associate Lagrange multiper from children set; Vj_x and associate Lagrange multiper from ancestor bus
        # In the x update, receive Pi_j_z, Qi_j_z and Ii_j_z from ancestor bus; and V_A_j_z from children set. In the x update, no multiper is required.

        Pji_x = modelX.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name="Pij")

    return Area


if __name__ == "__main__":
    from pypower import runopf

    mpc = case33.case33()  # Default test case
    (obj, residual) = run(mpc)

    result = runopf.runopf(case33.case33())

    gap = 100 * (result["f"] - obj) / obj

    print(gap)
