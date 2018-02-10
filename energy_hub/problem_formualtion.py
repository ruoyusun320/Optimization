"""
Problem formulation class of energy hub optimization
Two types of problems will be formulated.
1) Day-ahead optimization

2) Real-time optimization

@author: Tianyang Zhao
@e-mail: zhaoty@ntu.edu.sg
@date: 10 Feb 2018
"""
from numpy import zeros, ones, inf, vstack


class ProblemFormulation():
    """
    Two stage optimization problems formulation for energy hub systems
    """

    def __init__(self):
        self.name = 'EnergyHubProblem'

    def first_stage_problem(*args):
        # import data format
        from energy_hub.data_format import GAS, UG, PAC2DC, PDC2AC, PHVAC, EESS, PESSDC, PESSCH, NX
        # Parameter announcement
        PHVAC_max = args[0]
        eff_HVAC = args[1]
        Pess_ch_max = args[2]
        Pess_dc_max = args[3]
        eff_dc = args[4]
        eff_ch = args[5]
        E0 = args[6]
        Emax = args[7]
        Emin = args[8]
        BIC_cap = args[9]
        Gmax = args[10]
        eff_BIC = args[11]
        eff_CHP_e = args[12]
        eff_CHP_h = args[13]

        # Load profile
        AC_PD = args[14]
        DC_PD = args[15]
        HD = args[16]
        CD = args[17]
        PV_PG = args[18]

        # Price profile
        Gas_price = args[18]
        Electric_price = args[19]
        Eess_cost = args[20]

        Delta_t = args[21]
        T = args[22]
        # Formulate the boundary information
        lb = zeros((NX * T, 1))
        ub = zeros((NX * T, 1))
        for i in range(T):
            lb[i * NX + UG] = 0
            lb[i * NX + GAS] = 0
            lb[i * NX + PAC2DC] = 0
            lb[i * NX + PDC2AC] = 0
            lb[i * NX + EESS] = Emin
            lb[i * NX + PESSDC] = Pess_dc_max
            lb[i * NX + PESSCH] = Pess_ch_max
            lb[i * NX + PHVAC] = 0

            ub[i * NX + UG] = inf
            ub[i * NX + GAS] = Gmax
            ub[i * NX + PAC2DC] = BIC_cap
            ub[i * NX + PDC2AC] = BIC_cap
            ub[i * NX + EESS] = Emax
            ub[i * NX + PESSDC] = Pess_dc_max
            ub[i * NX + PESSCH] = Pess_ch_max
            ub[i * NX + PHVAC] = PHVAC_max
        # The constain set, all equal constraints
        Aeq = zeros((T, NX * T))
        beq = zeros((T, 1))
        for i in range(T):
            Aeq[i, i * NX + GAS] = eff_CHP_h
            beq[i] = HD[i]

        Aeq_temp = zeros((T, NX * T))
        beq_temp = zeros((T, 1))
        for i in range(T):
            Aeq_temp[i, i * NX + PHVAC] = eff_HVAC
            beq_temp[i] = CD[i]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = vstack([beq, beq_temp])

        Aeq_temp = zeros((T, NX * T))
        beq_temp = zeros((T, 1))
        for i in range(T):
            Aeq_temp[i, i * NX + UG] = 1
            Aeq_temp[i, i * NX + GAS] = eff_CHP_e
            Aeq_temp[i, i * NX + PDC2AC] = eff_BIC
            Aeq_temp[i, i * NX + PAC2DC] = -1
            beq_temp[i] = AC_PD[i]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = vstack([beq, beq_temp])

        Aeq_temp = zeros((T, NX * T))
        beq_temp = zeros((T, 1))
        for i in range(T):
            Aeq_temp[i, i * NX + PESSDC] = 1
            Aeq_temp[i, i * NX + PESSCH] = -1
            Aeq_temp[i, i * NX + PAC2DC] = eff_BIC
            Aeq_temp[i, i * NX + PDC2AC] = -1
            beq_temp[i] = DC_PD[i] - PV_PG[i]

        Aeq = vstack([Aeq, Aeq_temp])
        beq = vstack([beq, beq_temp])

        Aeq_temp = zeros((T, NX * T))
        beq_temp = zeros((T, 1))
        for i in range(T):
            if i == 1:
                Aeq_temp[i, i * NX + EESS] = 1
                Aeq_temp[i, i * NX + PESSCH] = -eff_ch * Delta_t
                Aeq_temp[i, i * NX + PESSDC] = eff_dc * Delta_t
                beq_temp[i] = E0
            else:
                Aeq_temp[i, i * NX + EESS] = 1
                Aeq_temp[i, (i - 1) * NX + EESS] = -1
                Aeq_temp[i, i * NX + PESSCH] = -eff_ch * Delta_t
                Aeq_temp[i, i * NX + PESSDC] = eff_dc * Delta_t
                beq_temp[i] = E0
        Aeq = vstack([Aeq, Aeq_temp])
        beq = vstack([beq, beq_temp])

        c = zeros((NX * T, 1))
        for i in range(T):
            c[i * NX + UG] = Electric_price
            c[i * NX + GAS] = Gas_price
            c[i * NX + PESSDC] = Eess_cost
            c[i * NX + PESSCH] = Eess_cost

        mathematical_model = {'c': c,
                              'Aeq': Aeq,
                              'beq': beq,
                              'A': None,
                              'b': None,
                              'lb': lb,
                              'ub': ub}
        return mathematical_model
