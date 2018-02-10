"""
Benders decomposition method for two-stage stochastic optimization problems
    Minimize a function F(X) beginning, subject to
	optional linear and nonlinear constraints and variable bounds:
			min  c'*x + sum(p_s*Q_s(x))
			x
			s.t. A*x<=b,
			     Aeq*x==beq,   x \in [lb,ub]
			where Q_s(x)=min q_s'*y
			             y
			             s.t. W_s*y = h_s-T_s*x
			             y \in R^+
References:
    [1]Benders Decomposition for Solving Two-stage Stochastic Optimization Models
    https://www.ima.umn.edu/materials/2015-2016/ND8.1-12.16/25378/Luedtke-spalgs.pdf
@author: Tianyang Zhao
@e-mail: zhaoty@ntu.edu.sg
@date: 8 Feb 2018
@version: 0.1

notes:
1) The data structure is based on the numpy and scipy
2) This algorithm should be extended for further version to solve the jointed chance constrained stochastic programming
3) In this test algorithm, Mosek is adpoted. https://www.mosek.com/
4) In the second stage optimization, the dual problem is solved.
5) The unbounded ray is obtained using Mosek as well.
"""


def benders_decompostion(c, A, b, Aeq, beq, lb, ub, q_s, W_s, h_s, T_s, Itermax, Gapmax):
    """
    Benders decomposition for two-stage stochastic programming
    :param c: The objective function in the first-stage in the first stage
    :param A: Inequality constrain coefficient matrix in the first stage
    :param b: Inequality constrain on the right side in the first stage
    :param Aeq: Equality constrain coefficient matrix in the first stage
    :param beq: Equality constrain on the right side in the first stage
    :param lb: Lower boundary in the first stage
    :param ub: Upper boundary in the first stage
    :param q_s: Objective function in the second stage
    :param W_s: Equality constrain coefficient matrix in the second stage
    :param h_s: Equality constrain in the second stage
    :param T_s: Equality constrain coefficient matrix in the second stage related to x
    :param Itermax: Equality constrain coefficient matrix in the second stage related to x
    :param Gapmax: Equality constrain coefficient matrix in the second stage related to x
    :return: The first stage solution and second stage solution
    """
    # Step 1:
    # 1) Input parameter check
    nx = len(c)
    ny = len(q_s)
    ns = W_s.size(0)
    # 2）

