"""This module implements a poisson cognitive hierarchy solver for the set of 
2x2 game matrices. The solver is able to solve pch for a series of poisson 
parameter value and plot the resulting p and q curve.

This model is implemmented from the paper "A Cognitive Hierarchy Model of Games"
by Camerer, Ho, and Weigelt (2005).
"""

import numpy as np
import math
import solvers.utils as utils


def poisson(k, t):
    """Returns the poisson probability of k given poisson parameter t"""
    return ((t ** k) * np.exp(-t)) / math.factorial(k)


def g_k(h, k, t):
    """Returns a k-level players' perceived probability of other players 
       thinking at level h given poisson parameter t"""
    return (poisson(h, t) / sum([poisson(l, t) for l in range(0, k)]))


def pch(game, top_k, t):
    """Returns the p and q values for a given game, poisson parameter t, and 
       maximum number of levels of thinking top_k"""
    p_s = [(0.5, 0.5)]
    q_s = [(0.5, 0.5)]
    for k in range(1, top_k):

        # Compute row probabilities for level k
        E_R_0 = sum([game.get_row_matrix()[0][c] * sum([g_k(h, k, t) * q_s[h][c]
                     for h in range(0, k)]) for c in range(0, 2)])
        E_R_1 = sum([game.get_row_matrix()[1][c] * sum([g_k(h, k, t) * q_s[h][c]
                        for h in range(0, k)]) for c in range(0, 2)])
        if E_R_0 > E_R_1:
            p_s.append((1, 0))
        elif E_R_0 < E_R_1:
            p_s.append((0, 1))
        else:
            p_s.append((0.5, 0.5))

        # Compute column probabilities for level k
        E_C_0 = sum([game.get_col_matrix()[r][0] * sum([g_k(h, k, t) * p_s[h][r]
                     for h in range(0, k)]) for r in range(0, 2)])
        E_C_1 = sum([game.get_col_matrix()[r][1] * sum([g_k(h, k, t) * p_s[h][r]
                        for h in range(0, k)]) for r in range(0, 2)])
        if E_C_0 > E_C_1:
            q_s.append((1, 0))
        elif E_C_0 < E_C_1:
            q_s.append((0, 1))
        else:
            q_s.append((0.5, 0.5))

    # Compute and return p and q based on distribution
    p = sum([p_s[i][0] * poisson(i, t) for i in range(0, top_k)])
    q = sum([q_s[i][0] * poisson(i, t) for i in range(0, top_k)])
    return p, q
    

def pch_curve(game, top_k, t_bot, t_top, step):
    """Collect the pch solutions for a range of poisson parameter values from  
    t_bot to t_top with step size step and return the resulting p and q values
    as numpy arrays"""
    p = np.array([])
    q = np.array([])
    t = np.arange(t_bot, t_top, step)
    for i in t:
        a, b = pch(game, top_k, i)
        p = np.append(p, a)
        q = np.append(q, b)
    return (p, q), t


def estimate_tau(game, sim_data):
    """Estimate the tau value for the poisson distribution on a given 
    game using simulated data values"""
    models, l = pch_curve(game, 10, 1, 5, 0.1)
    errors = np.array([utils.euclid_error((models[0][i], models[1][i]), sim_data) for i in range(len(models[0]))])
    return l[np.where(errors == min(errors))][0]


def pch_est(game, sim_data):
    """Estimate the pch solution for a given game using simulated data values"""
    tau = estimate_tau(game, sim_data)
    return pch(game, 10, tau), tau
