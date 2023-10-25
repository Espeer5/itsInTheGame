""" This module computes the quantal response equilibrium solution for a 2x2
Game object. The QRE solution is represented as the probability of playing
each strategy as a tuple of (row player p(row = 1), col player p(col = 1)) for 
a given lambda parameter."""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def qre(game, l):
    """ Computes a quantal response equilibrium solution for a 2x2 game for a given 
    lambda parameter, l. The QRE solution is represented as the probability of
    playing each strategy as a tuple of (row player p(row = 1), 
    col player p(col = 1)).
    """
    # Expectation of row i based on col player strategy q
    def E_R(i, q):
        pays = game.get_row_matrix()[i]
        return (q * pays[0] + (1 - q) * pays[1])
    
    # Expectation of col j based on row player strategy p
    def E_C(j, p):
        pays = [game.get_col_matrix()[i][j] for i in range(2)]
        return (p * pays[0] + (1 - p) * pays[1])

    # Equations to solve for the QRE solution
    def equations(vars):
        p, q = vars
        eq_1 = p - (1 / (1 + np.exp(l * (E_R(1, q) - E_R(0, q)))))
        eq_2 = q - (1 / (1 + np.exp(l * (E_C(1, p) - E_C(0, p)))))
        return [eq_1, eq_2]
    
    # Find and return optimal solution
    a, b = fsolve(equations, (0.01, 0.01))
    return a, b


def qre_curve(game, l_top, step):
    """Collect the qre solutions for a range of lambda values from 0 to l_top 
    with step size step and return the resulting p and q values as numpy 
    arrays"""
    p = np.array([])
    q = np.array([])
    lamb = np.arange(0, l_top, step)
    for l in lamb:
        a,b = qre(game, l)
        p = np.append(p, b)
        q = np.append(q, a)
    return p, q


def plot_qre(game, l_top, step):
    """Plot the qre curve for a range of lambda values from 0 to l_top 
    with step size step and save the resulting plot as qre.png, then 
    return the resulting p and q values as numpy arrays"""
    p, q = qre_curve(game, l_top, step)
    plt.plot(p, q, color='blue')
    plt.xlabel('p')
    plt.ylabel('q')
    plt.title('QRE Arc')
    plt.savefig('qre.png')
    return p, q