"""
Basic unit commitment to some mix-integer linear/quadratic programming problem
@author: Zhao Tianyang
@e-mail: zhaoty@ntu.edu.sg
@date:6 Mar 2018
"""


def main(case):
    """

    :param case: The test case for unit commitment problem
    :return:
    """
    from unit_commitment.data_format import IG, PG




    return model


if __name__ == "__main__":
    from pypower import case118

    test_case = case118.case118()
    model = main(test_case)
