"""
Benders decomposition method for two-stage stochastic optimization problems
    Minimize a function F(X) beginning, subject to
	optional linear and nonlinear constraints and variable bounds:
			min  c'x + sum(p_s*Q_s(x))
			x
			s.t. Ax>=b, x \in R
			where Q_s(x)=min q_s'y
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
1) The data structure is based on the
"""
