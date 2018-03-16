"""
To test the functions of unit commitment problems
"""
from pypower import case118
from pypower.runopf import runopf


runopf(case118.case118())