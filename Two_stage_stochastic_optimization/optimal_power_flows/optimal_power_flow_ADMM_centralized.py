"""
ADMM based distributed optimal power flow
The power flow modelling is based on the branch power flow

References:
    [1]Peng, Qiuyu, and Steven H. Low. "Distributed optimal power flow algorithm for radial networks, I: Balanced single phase case." IEEE Transactions on Smart Grid (2016).
    The centralized model (1)
"""

from Two_stage_stochastic_optimization.power_flow_modelling import case33
from pypower import runopf
from gurobipy import *
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, diag, concatenate, where
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack, diags
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
    i = range(nl)  ## double set of row indices
    # Connection matrix
    Cf = sparse((ones(nl), (i, f)), (nl, nb))
    Ct = sparse((ones(nl), (i, t)), (nl, nb))
    Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
    # Connection matrix
    Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
    Branch_R = branch[:, BR_R]
    Branch_X = branch[:, BR_X]
    Slmax = branch[:, RATE_A] / baseMVA
    gen[:, PMAX] = gen[:, PMAX] / baseMVA
    gen[:, PMIN] = gen[:, PMIN] / baseMVA
    gencost[:, 4] = gencost[:, 4] * baseMVA
    gencost[:, 5] = gencost[:, 5] * baseMVA
    bus[:, PD] = bus[:, PD] / baseMVA
    bus[:, QD] = bus[:, QD] / baseMVA
    area = ancestor_children_generation(f, t, range(nb),Branch_R,Branch_X, Slmax, gen, bus,gencost )
    # Formulate the centralized optimization problem according to the information provided by area
    model = Model("OPF")
    # Define the decision variables, compact set
    Pij_x = {}
    Qij_x = {}
    Iij_x = {}
    Vi_x = {}
    Pi_x = {}
    Qi_x = {}
    Pg = {}
    Qg = {}
    Pij_y = {}
    Qij_y = {}
    Iij_y = {}
    Vi_y = {}
    Pi_y = {}
    Qi_y = {}
    obj = 0
    for i in range(nb):# The iteration from each bus
        Pij_x[i] = model.addVar(lb=-area[i]["SMAX"], ub=area[i]["SMAX"], vtype=GRB.CONTINUOUS,
                                name="Pij_x{0}".format(i))
        Qij_x[i] = model.addVar(lb=-area[i]["SMAX"], ub=area[i]["SMAX"], vtype=GRB.CONTINUOUS,
                                name="Qij_x{0}".format(i))
        Iij_x[i] = model.addVar(lb=-area[i]["SMAX"], ub=area[i]["SMAX"], vtype=GRB.CONTINUOUS,
                                name="Iij_x{0}".format(i))
        Vi_x[i] = model.addVar(lb=area[i]["VMIN"], ub=area[i]["VMAX"], vtype=GRB.CONTINUOUS,
                               name="Vi_x{0}".format(i))
        Pi_x[i] = model.addVar(lb=area[i]["PGMIN"] - area[i]["PD"], ub=area[i]["PGMAX"] - area[i]["PD"],
                               vtype=GRB.CONTINUOUS,
                               name="Pi_x{0}".format(i))
        Qi_x[i] = model.addVar(lb=area[i]["QGMIN"] - area[i]["QD"], ub=area[i]["QGMAX"] - area[i]["PD"],
                               vtype=GRB.CONTINUOUS,
                               name="Qi_x{0}".format(i))
        Pg[i] = model.addVar(lb=area[i]["PGMIN"], ub=area[i]["PGMAX"], vtype=GRB.CONTINUOUS, name="Pi_x{0}".format(i))
        Qg[i] = model.addVar(lb=area[i]["QGMIN"], ub=area[i]["QGMAX"], vtype=GRB.CONTINUOUS, name="Qi_x{0}".format(i))

        Pij_y[i] = model.addVar(vtype=GRB.CONTINUOUS, name="Pij_y{0}".format(i))
        Qij_y[i] = model.addVar(vtype=GRB.CONTINUOUS, name="Qij_y{0}".format(i))
        Iij_y[i] = model.addVar(vtype=GRB.CONTINUOUS, name="Iij_y{0}".format(i))
        Vi_y[i] = model.addVar(vtype=GRB.CONTINUOUS, name="Vi_y{0}".format(i))
        Pi_y[i] = model.addVar(vtype=GRB.CONTINUOUS, name="Pi_y{0}".format(i))
        Qi_y[i] = model.addVar(vtype=GRB.CONTINUOUS, name="Qi_y{0}".format(i))

        if area[i]["TYPE"]=="ROOT":# If this bus is the root bus
            Pij_x[i] = model.addVar(lb=-area[i]["SMAX"], ub=area[i]["SMAX"], vtype=GRB.CONTINUOUS,
                                    name="Pij_x{0}".format(i))
            Qij_x[i] = model.addVar(lb=-area[i]["SMAX"], ub=area[i]["SMAX"], vtype=GRB.CONTINUOUS,
                                    name="Qij_x{0}".format(i))
            Iij_x[i] = model.addVar(lb=-area[i]["SMAX"], ub=area[i]["SMAX"], vtype=GRB.CONTINUOUS,
                                    name="Iij_x{0}".format(i))
            Vi_x[i] = model.addVar(lb=area[i]["VMIN"], ub=area[i]["VMAX"], vtype=GRB.CONTINUOUS,
                                    name="Vi_x{0}".format(i))
            Pi_x[i] = model.addVar(lb=area[i]["PGMIN"]-area[i]["PD"], ub=area[i]["PGMAX"]-area[i]["PD"], vtype=GRB.CONTINUOUS,
                                   name="Pi_x{0}".format(i))
            Qi_x[i] = model.addVar(lb=area[i]["QGMIN"] - area[i]["QD"], ub=area[i]["QGMAX"] - area[i]["PD"],
                                   vtype=GRB.CONTINUOUS,
                                   name="Qi_x{0}".format(i))
            Pg[i] = model.addVar(lb=area[i]["PGMIN"], ub=area[i]["PGMAX"],vtype=GRB.CONTINUOUS,name="Pi_x{0}".format(i))
            Qg[i] = model.addVar(lb=area[i]["QGMIN"], ub=area[i]["QGMAX"],vtype=GRB.CONTINUOUS,name="Qi_x{0}".format(i))
            Pij_y[i] = model.addVar( vtype=GRB.CONTINUOUS,name="Pij_y{0}".format(i))
            Qij_y[i] = model.addVar( vtype=GRB.CONTINUOUS,name="Qij_y{0}".format(i))
            Iij_y[i] = model.addVar( vtype=GRB.CONTINUOUS,name="Iij_y{0}".format(i))
            Vi_y[i] = model.addVar( vtype=GRB.CONTINUOUS,name="Vi_y{0}".format(i))
            Pi_y[i] = model.addVar(vtype=GRB.CONTINUOUS,name="Pi_y{0}".format(i))
            Qi_y[i] = model.addVar(vtype=GRB.CONTINUOUS,name="Qi_y{0}".format(i))





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
        primal_residual.append(Pij[i] * Pij[i] + Qij[i] * Qij[i] - Iij[i] * Vi[int(t[i])])

    return obj, primal_residual


def turn_to_power(list, power=1):
    return [number ** power for number in list]


def ancestor_children_generation(branch_f, branch_t, index, Branch_R, Branch_X, SMAX, gen, bus, gencost):
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
    Area = [ ]
    for i in index:
        temp = {}
        temp["Index"] = i
        if i in branch_t:
            AncestorBus = branch_f[where(branch_t == i)]
            temp["Ai"] = int(AncestorBus[0]) # For each bus, there exits only one ancestor bus, as one connected tree
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
            temp["Ai"] = [ ]
            temp["Abranch"] = [ ]
            temp["BR_R"] = 0
            temp["BR_X"] = 0
            temp["SMAX"] = 0
            temp["TYPE"] = "ROOT"

        if i in branch_f:
            ChildrenBranch = where(branch_f == i)
            nChildren = len(ChildrenBranch[0])
            temp["Ci"] = [ ]
            temp["Cbranch"] = [ ]
            for i in range(nChildren):
                temp["Cbranch"].append(int(ChildrenBranch[0][i]))
                temp["Ci"].append(int(branch_t[temp["Cbranch"][i]]))# The children bus
        else:
            temp["Cbranch"] = []
            temp["Ci"] = []

        # Update the node information
        if i in gen[:,GEN_BUS]:
            temp["PGMAX"] = gen[where(gen[:, GEN_BUS] == i), PMAX][0][0]
            temp["PGMIN"] = gen[where(gen[:, GEN_BUS] == i), PMIN][0][0]
            temp["QGMAX"] = gen[where(gen[:, GEN_BUS] == i), QMAX][0][0]
            temp["QGMIN"] = gen[where(gen[:, GEN_BUS] == i), QMIN][0][0]
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
        temp["VMIN"] = bus[i, VMIN]**2
        temp["VMAX"] = bus[i, VMAX]**2

        Area.append(temp)

    return Area


if __name__ == "__main__":
    from pypower import runopf

    mpc = case33.case33()  # Default test case
    (obj, residual) = run(mpc)

    result = runopf.runopf(case33.case33())

    gap = 100 * (result["f"] - obj) / obj

    print(gap)