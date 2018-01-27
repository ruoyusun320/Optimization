"""
Two stage stochastic optimization problem for the hybrid AC/DC microgrid embedded energy hub
@author: Tianyang Zhao
@mail: zhaoty@ntu.edu.sg
@date:27 Jan 2018
"""
from numpy import array,arange
from matplotlib import pyplot
from scipy import interpolate

def problem_formulation(N, delta, weight_factor):
    """
    Jointed optimization for the electrical and thermal optimisation
    :param N: number of scenario
    :param delta: forecasting errors
    :param weight_factor: weight factor between the first stage decision making and second stage decision makin
    :return:
    """
    # Parameters settings for the
    PHVDC_max = 50
    PV_cap = 20
    AC_PD_cap = 10
    DC_PD_cap = 10
    HD_cap = 5
    CD_cap = 5
    T_first_stage = 24

    # AC electrical demand
    AC_PD = array([323.0284, 308.2374, 318.1886, 307.9809, 331.2170, 368.6539, 702.0040, 577.7045, 1180.4547, 1227.6240,
                   1282.9344, 1311.9738, 1268.9502, 1321.7436, 1323.9218, 1327.1464, 1386.9117, 1321.6387, 1132.0476,
                   1109.2701, 882.5698, 832.4520, 349.3568, 299.9920])
    # DC electrical demand
    DC_PD = array([287.7698, 287.7698, 287.7698, 287.7698, 299.9920, 349.3582, 774.4047, 664.0625, 1132.6996, 1107.7366,
                   1069.6837, 1068.9819, 1027.3295, 1096.3820, 1109.4778, 1110.7039, 1160.1270, 1078.7839, 852.2514,
                   791.5814, 575.4085, 551.1441, 349.3568, 299.992])
    # Heating demand
    HD = array([16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
                86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
                0, 0])
    # Cooling demand
    CD = array([16.0996, 17.7652, 21.4254, 20.2980, 19.7012, 21.5134, 860.2167, 522.1926, 199.1072, 128.6201, 104.0959,
                86.9985, 95.0210, 59.0401, 42.6318, 26.5511, 39.2718, 73.3832, 120.9367, 135.2154, 182.2609, 201.2462,
                0, 0])
    # PV load profile
    PV_PG = array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.05, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35,
         0.14, 0.02, 0.02, 0.00, 0.00, 0.00])
    PV_PG = PV_PG*PV_cap
    # Modify the first stage profiles
    AC_PD = AC_PD / 2
    DC_PD = DC_PD / 2
    AC_PD = (AC_PD / max(AC_PD)) * AC_PD_cap
    DC_PD = (DC_PD / max(DC_PD)) * DC_PD_cap
    HD = (HD / max(HD)) * HD_cap
    CD = (CD / max(CD)) * CD_cap

    # Generate the second stage profiles using spline of scipy
    Time_first_stage = arange(0,T_first_stage,1)
    Time_second_stage = arange(0,T_first_stage,0.25)

    AC_PD_tck = interpolate.splrep(Time_first_stage, AC_PD, s=0)
    DC_PD_tck = interpolate.splrep(Time_first_stage, DC_PD, s=0)
    HD_tck = interpolate.splrep(Time_first_stage, HD, s=0)
    CD_tck = interpolate.splrep(Time_first_stage, CD, s=0)
    PV_PG_tck = interpolate.splrep(Time_first_stage, PV_PG, s=0)

    AC_PD_second_stage = interpolate.splev(Time_second_stage, AC_PD_tck, der=0)
    DC_PD_second_stage = interpolate.splev(Time_second_stage, DC_PD_tck, der=0)
    HD_second_stage = interpolate.splev(Time_second_stage, HD_tck, der=0)
    CD_second_stage = interpolate.splev(Time_second_stage, CD_tck, der=0)
    PV_PG_second_stage = interpolate.splev(Time_second_stage, PV_PG_tck, der=0)

    pyplot.plot(Time_first_stage, AC_PD, 'x', Time_second_stage, AC_PD_second_stage, 'b')
    pyplot.plot(Time_first_stage, DC_PD, 'x', Time_second_stage, DC_PD_second_stage, 'b')
    pyplot.plot(Time_first_stage, HD, 'x', Time_second_stage, HD_second_stage, 'b')
    pyplot.plot(Time_first_stage, CD, 'x', Time_second_stage, CD_second_stage, 'b')
    pyplot.plot(Time_first_stage, PV_PG, 'x', Time_second_stage, PV_PG_second_stage, 'b')

    pyplot.show()

    model = {}
    return model  # Formulated mixed integer linear programming problem


if __name__ == "__main__":
    model = problem_formulation(50, 0.03, 0)
