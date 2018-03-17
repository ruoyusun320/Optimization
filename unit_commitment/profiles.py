"""
The profiles which can be used in the unit commitment problems
1) Load profiles
2) Wind power profiles
3) PV profiles
"""
from numpy import array

PD_PROFILE = array(
    [0.233,
     0.222,
     0.229,
     0.222,
     0.239,
     0.266,
     0.506,
     0.416,
     0.851,
     0.885,
     0.925,
     0.946,
     0.915,
     0.953,
     0.954,
     0.957,
     1.000,
     0.953,
     0.816,
     0.799,
     0.636,
     0.600,
     0.259,
     0.216])

PV_PROFILE = array(
    [0.00,
     0.00,
     0.00,
     0.00,
     0.00,
     0.00,
     0.03,
     0.05,
     0.17,
     0.41,
     0.63,
     0.86,
     0.94,
     1.00,
     0.95,
     0.81,
     0.59,
     0.35,
     0.14,
     0.02,
     0.02,
     0.00,
     0.00,
     0.00])

