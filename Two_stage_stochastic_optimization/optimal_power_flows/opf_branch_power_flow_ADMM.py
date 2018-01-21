"""
ADMM based distributed optimal power flow
The power flow modelling is based on the branch power flow

References:
    [1]Peng, Qiuyu, and Steven H. Low. "Distributed optimal power flow algorithm for radial networks, I: Balanced single phase case." IEEE Transactions on Smart Grid (2016).
    [2]
"""

from Two_stage_stochastic_optimization.power_flow_modelling import case33
from pypower import runopf
from gurobipy import *
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, diag, concatenate, where
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack, diags


def run(mpc):
    """
    Gurobi based optimal power flow modelling and solution
    :param mpc: The input case of optimal power flow
    :return: obtained solution
    """
    # Data format
    from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
    from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
    from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I, QD
    from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN
    from pypower.ext2int import ext2int

    mpc = ext2int(mpc)
    baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

    nb = shape(mpc['bus'])[0]  # number of buses
    nl = shape(mpc['branch'])[0]  # number of branches
    ng = shape(mpc['gen'])[0]  # number of dispatchable injections

    for i in range(nl):  # branch indexing exchange
        if branch[i, F_BUS] > branch[i, T_BUS]:
            temp = branch[i, F_BUS]
            branch[i, F_BUS] = branch[i, T_BUS]
            branch[i, T_BUS] = temp

    f = branch[:, F_BUS]  ## list of "from" buses
    t = branch[:, T_BUS]  ## list of "to" buses
    area = ancestor_children_generation(f, t, range(nb))

    # Connection matrix
    Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
    Branch_R = branch[:, BR_R]
    Branch_X = branch[:, BR_X]

    # Obtain the boundary information

    Slmax = branch[:, RATE_A] / baseMVA

    Pij_l = -Slmax
    Qij_l = -Slmax
    Iij_l = zeros(nl)
    Vm_l = turn_to_power(bus[:, VMIN], 2)
    Pg_l = gen[:, PMIN] / baseMVA
    Qg_l = gen[:, QMIN] / baseMVA
    Pi_l = -bus[:, PD] / baseMVA + Cg * Pg_l
    Qi_l = -bus[:, QD] / baseMVA + Cg * Qg_l

    Pij_u = Slmax
    Qij_u = Slmax
    Iij_u = Slmax
    Vm_u = turn_to_power(bus[:, VMAX], 2)
    Pg_u = 2 * gen[:, PMAX] / baseMVA
    Qg_u = 2 * gen[:, QMAX] / baseMVA
    Pi_u = -bus[:, PD] / baseMVA + Cg * Pg_u
    Qi_u = -bus[:, QD] / baseMVA + Cg * Qg_u
    #
    model = Model("OPF")
    # Define the decision variables, compact set
    Pij = {}
    Qij = {}
    Iij = {}
    Vi = {}
    Pg = {}
    Qg = {}
    Pi = {}
    Qi = {}

    for i in range(nl):
        Pij[i] = model.addVar(lb=Pij_l[i], ub=Pij_u[i], vtype=GRB.CONTINUOUS, name="Pij{0}".format(i))
        Qij[i] = model.addVar(lb=Qij_l[i], ub=Qij_u[i], vtype=GRB.CONTINUOUS, name="Qij{0}".format(i))
        Iij[i] = model.addVar(lb=Iij_l[i], ub=Iij_u[i], vtype=GRB.CONTINUOUS, name="Iij{0}".format(i))

    for i in range(nb):
        Vi[i] = model.addVar(lb=Vm_l[i], ub=Vm_u[i], vtype=GRB.CONTINUOUS, name="V{0}".format(i))

    for i in range(ng):
        Pg[i] = model.addVar(lb=Pg_l[i], ub=Pg_u[i], vtype=GRB.CONTINUOUS, name="Pg{0}".format(i))
        Qg[i] = model.addVar(lb=Qg_l[i], ub=Qg_u[i], vtype=GRB.CONTINUOUS, name="Qg{0}".format(i))
    for i in range(nb):
        Pi[i] = model.addVar(lb=Pi_l[i], ub=Pi_u[i], vtype=GRB.CONTINUOUS, name="Pi{0}".format(i))
        Qi[i] = model.addVar(lb=Qi_l[i], ub=Qi_u[i], vtype=GRB.CONTINUOUS, name="Qi{0}".format(i))
    # For each area, before decomposition
    # Add system level constraints
    for i in range(nb):
        # If the bus is the root bus, only the children information is required.
        if len(area[i]["Ai"]) == 0:
            expr = 0
            for j in range(len(area[i]["Cbranch"][0])):
                expr += Pij[area[i]["Cbranch"][0][j]]

            model.addConstr(lhs=expr - Pi[i], sense=GRB.EQUAL, rhs=0)

            expr = 0
            for j in range(len(area[i]["Cbranch"][0])):
                expr += Qij[area[i]["Cbranch"][0][j]]

            model.addConstr(lhs=expr - Qi[i], sense=GRB.EQUAL, rhs=0)

        elif len(area[i]["Cbranch"]) == 0:  # This bus is the lead node

            model.addConstr(lhs=Pij[area[i]["Abranch"][0][0]] - Iij[area[i]["Abranch"][0][0]] * Branch_R[area[i]["Abranch"][0][0]] +
                    Pi[i],sense=GRB.EQUAL, rhs=0)

            model.addConstr(lhs=Qij[area[i]["Abranch"][0][0]] - Iij[area[i]["Abranch"][0][0]] * Branch_X[area[i]["Abranch"][0][0]] +
                    Qi[i],sense=GRB.EQUAL, rhs=0)

            model.addConstr(lhs=Vi[int(area[i]["Ai"][0])] - Vi[i] - 2 * Branch_R[area[i]["Abranch"][0][0]] * Pij[area[i]["Abranch"][0][0]] - 2 * Branch_X[area[i]["Abranch"][0][0]] * Qij[area[i]["Abranch"][0][0]] +
                                Iij[area[i]["Abranch"][0][0]] * (Branch_R[area[i]["Abranch"][0][0]] ** 2 + Branch_X[area[i]["Abranch"][0][0]] ** 2),sense=GRB.EQUAL, rhs=0)

            model.addConstr(
                Pij[area[i]["Abranch"][0][0]] * Pij[area[i]["Abranch"][0][0]] + Qij[area[i]["Abranch"][0][0]] * Qij[
                    area[i]["Abranch"][0][0]] <= Vi[int(area[i]["Ai"][0])] *
                Iij[area[i]["Abranch"][0][0]])

        else:
            expr = 0
            for j in range(len(area[i]["Cbranch"][0])):
                expr += Pij[area[i]["Cbranch"][0][j]]
            model.addConstr( lhs=
                Pij[area[i]["Abranch"][0][0]] - Iij[area[i]["Abranch"][0][0]] * Branch_R[area[i]["Abranch"][0][0]] +
                    Pi[i] - expr,sense=GRB.EQUAL, rhs=0)

            expr = 0
            for j in range(len(area[i]["Cbranch"][0])):
                expr += Qij[area[i]["Cbranch"][0][j]]

            model.addConstr(Qij[area[i]["Abranch"][0][0]] - Iij[area[i]["Abranch"][0][0]] * Branch_X[area[i]["Abranch"][0][0]] +
                    Qi[i] - expr,sense=GRB.EQUAL, rhs=0)

            model.addConstr(lhs=Vi[int(area[i]["Ai"][0])] - Vi[i] - 2 * Branch_R[area[i]["Abranch"][0][0]] * Pij[area[i]["Abranch"][0][0]] - 2 * Branch_X[area[i]["Abranch"][0][0]] * Qij[area[i]["Abranch"][0][0]] +
                                Iij[area[i]["Abranch"][0][0]] * (Branch_R[area[i]["Abranch"][0][0]] ** 2 + Branch_X[area[i]["Abranch"][0][0]] ** 2),sense=GRB.EQUAL, rhs=0)
            model.addConstr(
                Pij[area[i]["Abranch"][0][0]] * Pij[area[i]["Abranch"][0][0]] + Qij[area[i]["Abranch"][0][0]] * Qij[area[i]["Abranch"][0][0]] <= Vi[int(area[i]["Ai"][0])] *
                Iij[area[i]["Abranch"][0][0]])
    obj = 0
    for i in range(ng):
        model.addConstr(lhs=Pg[i] - Pi[int(gen[i, GEN_BUS])] ,sense=GRB.EQUAL, rhs= bus[int(gen[i, GEN_BUS]), PD] / baseMVA)
        model.addConstr(lhs= Qg[i] - Qi[int(gen[i, GEN_BUS])], sense=GRB.EQUAL, rhs= bus[int(gen[i, GEN_BUS]), QD] / baseMVA)
        obj += gencost[i, 4] * Pg[i] * Pg[i] * baseMVA * baseMVA + gencost[i, 5] * Pg[i] * baseMVA + gencost[i, 6]

    model.setObjective(obj)
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0
    model.Params.DisplayInterval = 1
    model.optimize()

    Pij = []
    Qij = []
    Iij = []
    Vi = []
    Pg = []
    Qg = []
    Pi = []
    Qi = []

    for i in range(nl):
        Pij.append(model.getVarByName("Pij{0}".format(i)).X)
        Qij.append(model.getVarByName("Qij{0}".format(i)).X)
        Iij.append(model.getVarByName("Iij{0}".format(i)).X)

    for i in range(nb):
        Vi.append(model.getVarByName("V{0}".format(i)).X)
        Pi.append(model.getVarByName("Pi{0}".format(i)).X)
        Qi.append(model.getVarByName("Qi{0}".format(i)).X)

    for i in range(ng):
        Pg.append(model.getVarByName("Pg{0}".format(i)).X)
        Qg.append(model.getVarByName("Qg{0}".format(i)).X)

    obj = obj.getValue()


    primal_residual = []

    for i in range(nl):
        primal_residual.append(Pij[i] * Pij[i] + Qij[i] * Qij[i] - Iij[i] * Vi[int(f[i])])

    return obj, primal_residual


def turn_to_power(list, power=1):
    return [number ** power for number in list]


def ancestor_children_generation(branch_f, branch_t, index):
    """
    Ancestor and children information for each node
    :param branch_f:
    :param branch_t:
    :param index: Bus index
    :return: Area, ancestor bus, children buses, line among buses
    """
    Area = []
    for i in index:
        temp = { }
        temp["Index"] = i
        if i in branch_t:
            temp["Ai"] = branch_f[where(branch_t == i)] # For each bus, there exits only one ancestor bus, as one connected tree
            temp["Abranch"] = where(branch_t == i)
            temp["Type"] = "NORM"
        else:
            temp["Ai"] = [ ]
            temp["Abranch"] = [ ]
            temp["Type"] = "ROOT"

        if i in branch_f:
            temp["Cbranch"] = where(branch_f == i)
        else:
            temp["Cbranch"] = []

        Area.append(temp)

    return Area

def sub_problem( Index, Area, ru):
    """
    Sub-problem optimization for each area, where the area is defined one bus and together with the
    :param Index: Target area
    :param Area: Area connection information
    :param ru: Penalty factor for the second order constraints
    :return: Area, updated information
    """

    modelX = Model("sub_opf_x")  # Sub-optimal power flow x update
    modelY = Model("sub_opf_y")  # Sub-optimal power flow y update

    if Area[Index]["Type"] == "ROOT":# Only needs to meet the KCL equation
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
        V_Ci_z = [ ]
        Pi_Ci_x = [ ]
        Qi_Ci_x = [ ]
        li_Ci_x = [ ]
        mu_Pi_Ci_x = [ ]
        mu_Qi_Ci_x = [ ]
        mu_li_Ci_x = [ ]
        Line_R = []
        Line_X = []
        for i in range(nChildren):# Information exchange
            V_Ci_z.append(Area[Area[Index]["Ci"][i]]["V_Ai_z"])# Voltage magnitude stored in children set

            Pi_Ci_x = Pi_Ci_x.append(Area[Area[Index]["Ci"][i]]["Pi_x"]) # Active power from i to Ai stored in children set
            Qi_Ci_x = Qi_Ci_x.append(Area[Area[Index]["Ci"][i]]["Qi_x"])  # Reactive power from i to Ai stored in children set
            li_Ci_x = li_Ci_x.append(Area[Area[Index]["Ci"][i]]["li_x"])  # Current from i to Ai stored in children set
            mu_Pi_Ci_x = mu_Pi_Ci_x.append(Area[Area[Index]["Ci"][i]]["mu_Pi_x"])  # Multiplier of active power from i to Ai stored in children set
            mu_Qi_Ci_x = mu_Qi_Ci_x.append(Area[Area[Index]["Ci"][i]]["mu_Qi_x"])  # Multiplier of reactive power from i to Ai stored in children set
            mu_li_Ci_x = mu_li_Ci_x.append(Area[Area[Index]["Ci"][i]]["mu_li_x"])  # Multiplier of current from i to Ai stored in children set
            Line_R = Line_R.append(Area[Area[Index]["Ci"][i]]["Line_R"]) # Line resistance of the children area
            Line_X = Line_X.append(Area[Area[Index]["Ci"][i]]["Line_X"]) # Line reactance of the children area


        if len(Area["Gen"])!=0:
            pi_x = modelX.addVar(lb = Area[Index]["PMIN"] - Area[Index]["PD"], ub = Area[Index]["PMAX"] - Area[Index]["PD"],
                                 vtype = GRB.CONTINUOUS, name="pi")
            qi_x = modelX.addVar(lb = Area[Index]["QMIN"] - Area[Index]["QD"], ub = Area[Index]["QMAX"] - Area[Index]["QD"],
                                 vtype = GRB.CONTINUOUS, name="qi")
            Pg = modelX.addVar(lb = Area[Index]["PMIN"], ub = Area[Index]["PMAX"], vtype = GRB.CONTINUOUS, name="Pg")
            Qg = modelX.addVar(lb = Area[Index]["QMIN"], ub = Area[Index]["QMAX"], vtype = GRB.CONTINUOUS, name="Qg")

            pi_z = modelY.addVar(lb=Area[Index]["PMIN"] - Area[Index]["PD"], ub=Area[Index]["PMAX"] - Area[Index]["PD"],
                             vtype=GRB.CONTINUOUS, name="pi")
            qi_z = modelY.addVar(lb=Area[Index]["QMIN"] - Area[Index]["QD"], ub=Area[Index]["QMAX"] - Area[Index]["QD"],
                             vtype=GRB.CONTINUOUS, name="qi")

        else:
            pi_x = modelX.addVar(lb= - Area[Index]["PD"], ub= - Area[Index]["PD"], vtype=GRB.CONTINUOUS, name="Pi")
            qi_x = modelX.addVar(lb= - Area[Index]["QD"], ub= - Area[Index]["QD"], vtype=GRB.CONTINUOUS, name="Qi")
            Pg = modelX.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name="Pg")
            Qg = modelX.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name="Qg")
            pi_z = modelY.addVar(lb=- Area[Index]["PD"], ub=- Area[Index]["PD"], vtype=GRB.CONTINUOUS, name="Pi")
            qi_z = modelY.addVar(lb=- Area[Index]["QD"], ub=- Area[Index]["QD"], vtype=GRB.CONTINUOUS, name="Qi")

        Vi_x = modelX.addVar(lb=Area[Index]["VMIN"], ub=Area[Index]["VMAX"], vtype=GRB.CONTINUOUS, name="Vi")

        Pji_z = modelY.addVar(1,nChildren)
        Qji_z = modelY.addVar(1, nChildren)
        lji_z = modelY.addVar(1, nChildren)

        modelX.addConstr(Pg - pi_x == Area[Index]["PD"])
        modelX.addConstr(Qg - qi_x == Area[Index]["QD"])


        # KCL equations
        # 1) Active power balance equation
        expr = 0
        for i in range(nChildren):
            expr = expr + Pji_z[i] - lji_z[i]*Line_R[i]
        modelY.addConstr(lhs=expr + pi_z, sense=GRB.EQUAL, rhs=0)
        # 2) Reactive power balance equation
        expr = 0
        for i in range(nChildren):
            expr = expr + Qji_z[i] - lji_z[i] * Line_X[i]
        modelY.addConstr(lhs=expr + qi_z, sense=GRB.EQUAL, rhs=0)



    elif Area[Index]["Type"] == "LEAF": # Only needs to meet the KVL equation
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

    else: # Only needs to meet the KVL equation
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
